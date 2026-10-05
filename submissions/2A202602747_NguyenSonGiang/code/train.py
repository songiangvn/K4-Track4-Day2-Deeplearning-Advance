"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Một hàm `run(cfg)` dùng cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh (từ thư mục code/):
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc để cùng định nghĩa
với lúc chấm.

Lựa chọn cài đặt (ghi lại cho báo cáo):
  - LR cập nhật theo BƯỚC: warmup tuyến tính `warmup_epochs` rồi cosine về 0 ở bước cuối.
  - val loss luôn là cross-entropy thường (không smoothing/focal/trọng số) để so được giữa các loss.
  - EMA: trung bình cả tham số lẫn buffer BN (running_mean/var), decay có khởi động
    min(d, (1 + t) / (10 + t)); khi bật EMA, đánh giá và chọn checkpoint bằng trọng số EMA.
  - Tái lập: seed random/numpy/torch/worker cố định; cudnn.benchmark=True nên không bảo đảm
    giống từng bit giữa hai lần chạy (đổi lấy tốc độ).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch.multiprocessing

# Worker gửi tensor qua shared memory theo tên file thay vì giữ một file descriptor cho mỗi tensor
# (cách PyTorch khuyến nghị khi gặp "Too many open files").
torch.multiprocessing.set_sharing_strategy("file_system")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]          # submissions/<mssv>_<ten>/code -> gốc repo (chứa eval.py)
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    desc: str = ""                    # mô tả ngắn, dùng trong tên ảnh curves/<exp_id>_<desc>.png
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"        # tên timm, có thể kèm tag: "convnext_tiny.fb_in1k"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    drop_path_rate: float | None = None
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    crop_pct: float = 0.875           # val/test: Resize(img_size / crop_pct) + CenterCrop(img_size)
    aug: str = "basic"                # basic | color | vflip | trivial | randaug
    rrc_scale_min: float = 0.08       # RandomResizedCrop scale=(rrc_scale_min, 1); 0.08 = mặc định
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    optimizer: str = "adamw"          # adamw | sgd (momentum 0.9, nesterov)
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    grad_clip: float | None = None
    ema_decay: float | None = None
    # --- chưng cất tri thức (điểm thưởng): teacher = một lần chạy đã train, "<exp_id>/seed<k>" dưới out_dir ---
    teacher_run: str | None = None
    kd_alpha: float = 0.5             # loss = (1 - a) * loss_nhãn + a * T^2 * KL(teacher/T || student/T)
    kd_temperature: float = 4.0
    amp: bool = True
    channels_last: bool = True
    num_workers: int = 2
    cache: bool = True                # nạp byte JPEG từ data/images_bytes.pkl (dataset.load_byte_store)
    bench_latency: bool = True        # độ trễ sơ bộ batch 1 sau khi train (đo kỹ ở Bước 3)
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    name = f"{cfg.exp_id}_{cfg.desc}" if cfg.desc else cfg.exp_id
    return Path(cfg.curves_dir) / f"{name}.png"


def set_seed(seed: int) -> None:
    """Cố định random, numpy, torch (CPU và CUDA). Seed của worker đặt trong dataset.make_loader."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def build_optimizer(model, cfg: Config):
    """AdamW (hoặc SGD) với các nhóm tham số của model.param_groups."""
    import torch
    from model import param_groups
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(f"optimizer={cfg.optimizer!r} không hỗ trợ; chọn adamw hoặc sgd")


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Hệ số nhân LR ở bước `step` (0-based): warmup tuyến tính rồi cosine về 0."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0, cập nhật theo bước (slide trang 55)."""
    import torch
    total = cfg.epochs * steps_per_epoch
    warmup = int(round(cfg.warmup_epochs * steps_per_epoch))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total, warmup))


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    Giữ một bản sao riêng `self.module` để đánh giá. Buffer dạng float (running_mean/var của BN)
    cũng được trung bình như tham số; buffer nguyên (num_batches_tracked) được sao chép.
    Decay có khởi động: d_t = min(decay, (1 + t) / (10 + t)) để vài trăm bước đầu không bị
    kéo về trọng số khởi tạo.
    """

    def __init__(self, model, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    def update(self, model) -> None:
        import torch
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        with torch.no_grad():
            live = model.state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(live[k].detach(), alpha=1 - d)
                else:
                    v.copy_(live[k])


def kd_loss(student_logits, teacher_logits, T: float):
    """T^2 * KL(softmax(teacher/T) || softmax(student/T)), trung bình theo batch (Hinton et al. 2015)."""
    import torch.nn.functional as F
    return F.kl_div(F.log_softmax(student_logits.float() / T, -1), F.softmax(teacher_logits.float() / T, -1),
                    reduction="batchmean") * T * T


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None, teacher=None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "train_acc", "lrs": [lr head theo bước], ...}.

    train_acc là NaN khi dùng Mixup/CutMix (nhãn đã trộn, không có nghĩa).
    """
    import torch
    from losses import mix_batch, mixed_loss
    from model import set_train_mode

    set_train_mode(model)
    tot_loss, tot_correct, n, lrs = 0.0, 0, 0, []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        targets = None
        if cfg.mix:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.amp):
            logits = model(x)
            loss = mixed_loss(criterion, logits, targets) if targets else criterion(logits, y)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(x)
                loss = (1 - cfg.kd_alpha) * loss + cfg.kd_alpha * kd_loss(logits, t_logits, cfg.kd_temperature)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss = {loss.item()} (không hữu hạn)")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        lrs.append([g["lr"] for g in optimizer.param_groups])
        scheduler.step()
        if ema is not None:
            ema.update(model)
        bs = y.size(0)
        tot_loss += loss.item() * bs
        n += bs
        if not cfg.mix:
            tot_correct += (logits.argmax(1) == y).sum().item()
    return {"train_loss": tot_loss / n, "train_acc": tot_correct / n if not cfg.mix else float("nan"),
            "lrs": lrs}


def evaluate(model, loader, criterion, device, amp: bool = True, channels_last: bool = True):
    """Chạy model trên một loader ở chế độ eval, không gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9] float32, loss: float),
    giữ đúng thứ tự của loader. `criterion` dùng để tính loss (thường là CE thường).
    """
    import torch
    model.eval()
    names, ys, outs = [], [], []
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            outs.append(logits.float().cpu())
            ys.append(y)
            names.extend(f)
    logits = torch.cat(outs)
    y = torch.cat(ys)
    loss = float(criterion(logits, y))
    return names, y.numpy(), logits.numpy().astype(np.float32), loss


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str, lrs=None, steps_per_epoch=None) -> None:
    """Vẽ đường cong training -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    Ba ô: loss train/val; macro-F1 và top-1 val (và acc train nếu có); LR theo bước.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(ep, [h["train_loss"] for h in history], "o-", label="train loss (loss huấn luyện)")
    ax.plot(ep, [h["val_loss"] for h in history], "s-", label="val loss (CE)")
    ax.set_xlabel("epoch"), ax.set_ylabel("loss"), ax.set_title("Loss"), ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ep, [h["val_macro_f1"] for h in history], "o-", label="val macro-F1")
    ax.plot(ep, [h["val_top1"] for h in history], "s-", label="val top-1")
    if "val_macro_f1_live" in history[0]:
        ax.plot(ep, [h["val_macro_f1_live"] for h in history], "x--", label="val macro-F1 (không EMA)")
    tr = [h["train_acc"] for h in history]
    if not all(math.isnan(t) for t in tr):
        ax.plot(ep, tr, "^:", label="train acc")
    best = max(history, key=lambda h: (h["val_macro_f1"], -h["epoch"]))
    ax.axvline(best["epoch"], color="gray", ls=":", label=f"best epoch {best['epoch']}")
    ax.set_xlabel("epoch"), ax.set_ylabel("chỉ số"), ax.set_title("Chỉ số val"), ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    if lrs is not None and len(lrs):
        lrs = np.asarray(lrs)
        x = np.arange(len(lrs)) / (steps_per_epoch or 1)
        ax.plot(x, lrs.max(1), label="LR lớn nhất (head)")
        ax.plot(x, lrs.min(1), label="LR nhỏ nhất (backbone)")
        ax.set_yscale("log"), ax.legend(fontsize=8)
    ax.set_xlabel("epoch"), ax.set_ylabel("learning rate"), ax.set_title("LR theo bước")
    ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _metrics(y, logits) -> dict:
    from eval import compute_metrics
    probs = softmax_np(logits)
    m = compute_metrics(y, probs.argmax(1), probs)
    return {k: m[k] for k in ("top1", "macro_f1", "balanced_acc", "ece", "nll")} | \
           {"f1_per_class": m["f1"].tolist(), "recall_per_class": m["recall"].tolist()}


def close_loaders(loaders) -> None:
    """Dừng hẳn worker của các DataLoader (persistent_workers giữ chúng sống tới khi bị thu gom)."""
    import gc
    for dl in loaders:
        it = getattr(dl, "_iterator", None)
        if it is not None and hasattr(it, "_shutdown_workers"):
            it._shutdown_workers()
    loaders.clear()
    gc.collect()


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình (xem _run). Luôn dừng worker DataLoader khi kết thúc, kể cả khi lỗi.

    Lưu ý: với persistent_workers + pin_memory (torch 2.6), mỗi lần chạy vẫn để lại ~40 pipe sau khi
    dừng worker (đo bằng /proc/self/fd; tắt một trong hai thì không rò). Gọi run() nhiều lần trong một
    process sẽ dẫn tới "OSError: Too many open files", nên run_exps.py chạy MỖI thí nghiệm trong một
    process con riêng."""
    loaders: list = []
    try:
        return _run(cfg, loaders)
    finally:
        close_loaders(loaders)


def _run(cfg: Config, loaders: list) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    Lưu trong run_dir(cfg): config.json, history.csv, lr_steps.npy, best.pt, val_logits.npy,
    summary.json (và test_logits.npy nếu save_test_predictions). Dự đoán val (và test ở chung kết)
    ghi qua eval.save_predictions vào pred_dir. Test KHÔNG dùng cho bất kỳ quyết định nào (S4):
    chỉ đánh giá một lần sau khi đã chọn xong checkpoint bằng val.
    """
    import pandas as pd
    import timm
    import torch
    import torchvision
    from torch import nn

    import dataset as D
    from eval import save_predictions
    from losses import build_criterion, class_weights
    from model import build_model, count_gmacs, count_params

    set_seed(cfg.seed)
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and os.environ.get("REQUIRE_GPU") == "1":
        # job SLURM: node có GPU hỏng thì dừng ngay thay vì âm thầm train bằng CPU (cả ngày)
        raise RuntimeError("REQUIRE_GPU=1 nhưng torch không thấy GPU (node lỗi?)")
    env = {"torch": torch.__version__, "timm": timm.__version__, "torchvision": torchvision.__version__,
           "python": sys.version.split()[0],
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}

    # --- dữ liệu ---
    train_df, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    D.check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False)
    cache = D.load_byte_store(cfg.images_dir) if cfg.cache else False
    tf_train = D.build_transforms(True, cfg.img_size, cfg.aug, scale_min=cfg.rrc_scale_min)
    tf_eval = D.build_transforms(False, cfg.img_size, crop_pct=cfg.crop_pct)
    kw = dict(num_workers=cfg.num_workers, seed=cfg.seed, cache=cache)
    train_loader = D.make_loader(train_df, cfg.images_dir, tf_train, cfg.batch_size, True, cfg.sampler, **kw)
    val_loader = D.make_loader(val_df, cfg.images_dir, tf_eval, 128, False, **kw)
    loaders += [train_loader, val_loader]

    # --- mô hình, loss, tối ưu ---
    model = build_model(cfg.backbone, num_classes=D.NUM_CLASSES, drop_rate=cfg.drop_rate,
                        init=cfg.init, drop_path_rate=cfg.drop_path_rate, img_size=cfg.img_size)
    n_params, gmacs = count_params(model), count_gmacs(model, cfg.img_size)
    model = model.to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)

    counts = np.bincount(train_df["Label"], minlength=D.NUM_CLASSES)
    weight = None
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        weight = class_weights(counts, cfg.class_weight_beta or 0.0)
    if cfg.loss == "ls":
        criterion = build_criterion("ls", smoothing=cfg.label_smoothing)
    elif cfg.loss == "focal":
        criterion = build_criterion("focal", gamma=cfg.focal_gamma, alpha=weight)
    else:
        criterion = build_criterion(cfg.loss, weight=weight)
    criterion = criterion.to(device)
    val_criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    teacher = None
    if cfg.teacher_run:
        if cfg.mix:
            raise ValueError("KD chưa hỗ trợ kèm Mixup/CutMix")
        from inference import load_model
        teacher, _ = load_model(Path(cfg.out_dir) / cfg.teacher_run, device)
        if cfg.channels_last:
            teacher = teacher.to(memory_format=torch.channels_last)

    config = dataclasses.asdict(cfg) | {"weight_tag": model.weight_tag, "params_M": n_params,
                                        "gmacs": gmacs, "env": env, "train_counts": counts.tolist(),
                                        "class_weight": None if weight is None else weight.tolist()}
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"[{cfg.exp_id} seed{cfg.seed}] {cfg.backbone} ({model.weight_tag}) init={cfg.init} "
          f"params={n_params:.2f}M GMAC={gmacs:.2f} steps/epoch={len(train_loader)} on {env['gpu']}",
          flush=True)

    # --- huấn luyện, chọn checkpoint theo macro-F1 val (hòa: epoch sớm hơn) ---
    history, all_lrs, best_f1, best_epoch, best_state = [], [], -1.0, -1, None
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema,
                             teacher)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time = time.perf_counter() - t0
        all_lrs.extend(tr.pop("lrs"))

        eval_model = ema.module if ema else model
        _, yv, lv, val_loss = evaluate(eval_model, val_loader, val_criterion, device, cfg.amp, cfg.channels_last)
        mv = _metrics(yv, lv)
        row = {"epoch": epoch, **tr, "val_loss": val_loss, "val_macro_f1": mv["macro_f1"],
               "val_top1": mv["top1"], "val_balanced_acc": mv["balanced_acc"], "val_ece": mv["ece"],
               "lr_head_end": all_lrs[-1][-1] if all_lrs else None, "train_time_s": train_time,
               "epoch_time_s": time.perf_counter() - t0}
        if ema:
            _, _, ll, _ = evaluate(model, val_loader, val_criterion, device, cfg.amp, cfg.channels_last)
            row["val_macro_f1_live"] = _metrics(yv, ll)["macro_f1"]
        history.append(row)
        improved = mv["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch = mv["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
        print(f"  ep{epoch:02d} train_loss={tr['train_loss']:.4f} acc={tr['train_acc']:.4f} "
              f"val_loss={val_loss:.4f} F1={mv['macro_f1']:.4f} top1={mv['top1']:.4f} "
              f"({train_time:.0f}s){' *' if improved else ''}", flush=True)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    np.save(out / "lr_steps.npy", np.asarray(all_lrs, dtype=np.float32))
    plot_curves(history, curve_path(cfg),
                f"{cfg.exp_id} | {cfg.backbone} | {cfg.desc or ''} | seed {cfg.seed} "
                f"(best val macro-F1 {best_f1:.4f} @ ep {best_epoch})",
                all_lrs, len(train_loader))

    # --- nạp checkpoint tốt nhất, lưu logit/dự đoán val ---
    final = ema.module if ema else model
    final.load_state_dict(best_state)
    torch.save(best_state, out / "best.pt")
    names_v, yv, lv, _ = evaluate(final, val_loader, val_criterion, device, cfg.amp, cfg.channels_last)
    np.save(out / "val_logits.npy", lv)
    save_predictions(pred_path(cfg, "val"), names_v, yv, softmax_np(lv))
    mv = _metrics(yv, lv)

    summary = {"exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone,
               "weight_tag": model.weight_tag, "params_M": n_params, "gmacs": gmacs,
               "best_epoch": best_epoch, "val": mv,
               "train_time_per_epoch_s": float(np.mean([h["train_time_s"] for h in history])),
               "env": env}

    # --- chỉ ở Bước 4: test đúng một lần ---
    if cfg.save_test_predictions:
        test_loader = D.make_loader(test_df, cfg.images_dir, tf_eval, 128, False, **kw)
        loaders.append(test_loader)
        names_t, yt, lt, _ = evaluate(final, test_loader, val_criterion, device, cfg.amp, cfg.channels_last)
        np.save(out / "test_logits.npy", lt)
        save_predictions(pred_path(cfg, "test"), names_t, yt, softmax_np(lt))
        summary["test_predictions"] = str(pred_path(cfg, "test"))   # chỉ số test tính sau bằng eval.py

    if cfg.bench_latency and device.type == "cuda":
        from benchmark import latency_report
        final.to(memory_format=torch.contiguous_format)
        summary["latency_b1_fp32"] = latency_report(final, 1, cfg.img_size, "fp32", iters=50)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[{cfg.exp_id} seed{cfg.seed}] best epoch {best_epoch}: val macro-F1 {mv['macro_f1']:.4f} "
          f"top-1 {mv['top1']:.4f}", flush=True)
    return summary


def _cast(value: str, type_str: str):
    if value.lower() in ("none", "null") and "None" in type_str:
        return None
    base = type_str.replace("| None", "").strip()
    if base == "bool":
        if value.lower() in ("1", "true", "yes"):
            return True
        if value.lower() in ("0", "false", "no"):
            return False
        raise ValueError(f"Không đọc được bool từ {value!r}")
    if base == "int":
        return int(value)
    if base == "float":
        return float(value)
    return value


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    fields = {f.name: str(f.type) for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Thiếu '=' trong {pair!r} (dạng KEY=VALUE)")
        k, v = pair.split("=", 1)
        if k not in fields:
            raise KeyError(f"Config không có trường {k!r}. Các trường: {sorted(fields)}")
        out[k] = _cast(v, fields[k])
    return out


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
