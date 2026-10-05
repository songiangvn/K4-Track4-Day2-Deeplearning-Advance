"""robustness.py - lệch phân phối và thích ứng lúc kiểm tra (điểm thưởng, RUBRIC mục 2).

    python robustness.py --run F01/seed0 [--tag rob_F01] [--severities 1 2 3]

Tập lệch miền TỰ TẠO từ VAL fold 0 (không dùng test để thích ứng hay chọn gì):
  dark  : làm tối (nhân độ sáng 0.6 / 0.4 / 0.25) - thiếu sáng, bóng râm
  noise : nhiễu Gauss sigma 0.04 / 0.08 / 0.12 (trên ảnh [0, 1]) - cảm biến
  blur  : Gaussian blur sigma 1 / 2 / 3 px - rung, lệch nét
  jpeg  : nén JPEG chất lượng 30 / 15 / 8 - truyền ảnh băng thông thấp
Mức 0 = val sạch.

Với mỗi (loại, mức) đo macro-F1, top-1, ECE của:
  none     model gốc
  ts       model gốc + temperature scaling, T khớp trên VAL SẠCH (như lúc triển khai: không biết trước
           miền mới) -> T còn đáng tin khi lệch miền không? (GUIDE câu hỏi 8)
  bnadapt  ước lượng lại thống kê BatchNorm trên dữ liệu lệch miền, không nhãn (Schneider et al. 2020)
  tent     BN dùng thống kê batch + cập nhật affine của lớp chuẩn hoá (BN/LN) bằng cực tiểu entropy
           dự đoán, SGD lr 2.5e-4 momentum 0.9, batch 64, 1 lượt (thiết lập ImageNet của Wang et al. 2021)
Chống rò rỉ khi thích ứng: chia val lệch miền thành 2 nửa (phân tầng). Thích ứng trên nửa A (không
dùng nhãn), dự đoán nửa B và ngược lại; mọi ảnh đều được dự đoán bởi model chưa thấy chính nó.
Mạng không có BN (ConvNeXt, ViT, Swin): bnadapt không áp dụng, tent chỉ cập nhật affine của LayerNorm.
"""
from __future__ import annotations

import argparse
import copy
import io
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

import dataset as D
import inference as I
import run_inference as RI

LEVELS = {
    "dark": [0.6, 0.4, 0.25],
    "noise": [0.04, 0.08, 0.12],
    "blur": [1.0, 2.0, 3.0],
    "jpeg": [30, 15, 8],
}


class Corrupt:
    """Biến đổi ảnh PIL -> PIL theo loại và mức. Nhiễu lấy từ RNG seed theo nội dung ảnh (crc32), nên
    cùng một ảnh luôn nhận cùng một mẫu nhiễu ở mọi lượt chạy và mọi phương pháp."""

    def __init__(self, kind: str, level):
        self.kind, self.level = kind, level

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.kind == "jpeg":
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=int(self.level))
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        x = np.asarray(img, dtype=np.float32) / 255.0
        if self.kind == "dark":
            x = x * self.level
        elif self.kind == "noise":
            rng = np.random.default_rng(zlib.crc32(np.asarray(img).tobytes()))
            x = x + rng.normal(0, self.level, x.shape).astype(np.float32)
        elif self.kind == "blur":
            from PIL import ImageFilter
            return img.filter(ImageFilter.GaussianBlur(radius=self.level))
        return Image.fromarray((np.clip(x, 0, 1) * 255).round().astype(np.uint8))


def make_transform(kind: str | None, level, img_size=224, crop_pct: float = D.CROP_PCT):
    """Tiền xử lý val của lần chạy, chèn phép làm hỏng SAU khi resize/crop (trên ảnh 224)."""
    from torchvision import transforms as T
    ops = [T.Resize(round(img_size / crop_pct)), T.CenterCrop(img_size)]
    if kind:
        ops.append(Corrupt(kind, level))
    return T.Compose([*ops, T.ToTensor(), T.Normalize(D.IMAGENET_MEAN, D.IMAGENET_STD)])


def norm_layers(model):
    return [m for m in model.modules()
            if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm))]


def has_bn(model) -> bool:
    return any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())


@torch.no_grad()
def bn_adapt(model, loader, device):
    """Ước lượng lại running_mean/var của mọi BN trên `loader` (trung bình tích luỹ, không nhãn)."""
    model = copy.deepcopy(model)
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                    # None = trung bình tích luỹ trên toàn bộ dữ liệu
    model.eval()
    for m in bns:
        m.train()
    for x, _, _ in loader:
        model(x.to(device))
    return model.eval()


def tent(model, loader, device, lr: float = 2.5e-4, momentum: float = 0.9):
    """Tent theo thiết lập ImageNet của bài báo (Wang et al. 2021): BN dùng thống kê của chính batch
    (không running stats), cập nhật affine của các lớp chuẩn hoá bằng cực tiểu entropy, SGD lr 2.5e-4,
    momentum 0.9, batch 64, 1 lượt. Model trả về vẫn dùng thống kê batch khi dự đoán."""
    model = copy.deepcopy(model)
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False
            m.running_mean = m.running_var = None
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for m in norm_layers(model):
        for p in (m.weight, m.bias):
            if p is not None:
                p.requires_grad_(True)
                params.append(p)
    opt = torch.optim.SGD(params, lr=lr, momentum=momentum)
    model.eval()                             # BN không còn running stats nên luôn dùng thống kê batch
    for x, _, _ in loader:
        p = model(x.to(device)).float().softmax(-1)
        ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
        opt.zero_grad()
        ent.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def halves(y: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        h[idx] = np.arange(len(idx)) % 2
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--severities", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--kinds", nargs="*", default=list(LEVELS))
    ap.add_argument("--num-workers", type=int, default=10)
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = RI.Ctx(RI.RUNS / args.run, dev, args.num_workers)
    model = I.load_model(ctx.run, dev)[0].float().eval()
    val = ctx.splits["val"].reset_index(drop=True)
    half = halves(val["Label"].to_numpy())
    tag = args.tag or f"rob_{args.run.replace('/', '_')}"
    out = RI.RUNS / "robustness"
    out.mkdir(parents=True, exist_ok=True)

    def loader(df, kind, level):
        return D.make_loader(df, RI.IMAGES, make_transform(kind, level, img_size=ctx.img_size, crop_pct=ctx.crop_pct), 64, False,
                             num_workers=args.num_workers, cache=ctx.store, persistent=False, seed=0)

    # T khớp trên val sạch (giống lúc triển khai)
    _, y_clean, z_clean = I.predict_logits(model, loader(val, None, None), dev, amp=False)
    T_clean = I.fit_temperature(z_clean, y_clean)
    print(f"T (val sạch) = {T_clean:.4f}; có BN: {has_bn(model)}", flush=True)

    settings = [("clean", 0, None)] + [(k, s, LEVELS[k][s - 1]) for k in args.kinds for s in args.severities]
    rows = []
    for kind, sev, level in settings:
        kind_arg = None if kind == "clean" else kind
        _, y, z = I.predict_logits(model, loader(val, kind_arg, level), dev, amp=False)
        res = {"none": I.softmax(z), "ts": I.apply_temperature(z, T_clean)}
        for name, fn in (("bnadapt", bn_adapt), ("tent", tent)):
            if name == "bnadapt" and not has_bn(model):
                continue
            probs = np.empty_like(res["none"])
            for h in (0, 1):
                adapt_df, eval_df = val[half != h], val[half == h]
                adapted = fn(model, loader(adapt_df, kind_arg, level), dev)
                _, _, zh = I.predict_logits(adapted, loader(eval_df, kind_arg, level), dev, amp=False)
                probs[half == h] = I.softmax(zh)
                del adapted
            res[name] = probs
        for method, p in res.items():
            m = RI.metrics(y, p)
            rows.append({"corruption": kind, "severity": sev, "level": level, "method": method,
                         **{k: m[k] for k in ("macro_f1_val", "top1_val", "ece_val", "nll_val")},
                         "recall_chinee": m["recall_Chinee_Apple"], "recall_snake": m["recall_Snake_Weed"]})
            print(f"{kind:6s} s{sev} {method:8s} F1={m['macro_f1_val']:.4f} top1={m['top1_val']:.4f} "
                  f"ECE={m['ece_val']:.4f}", flush=True)
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows).assign(run=args.run, backbone=ctx.cfg["backbone"], T_clean=T_clean)
    df.to_csv(out / f"{tag}.csv", index=False)
    plot(df, RI.SUB / "figures" / f"{tag}.png", f"{args.run} ({ctx.cfg['backbone']})")
    print("Đã ghi", out / f"{tag}.csv")


def plot(df, path, title):
    kinds = [k for k in LEVELS if k in set(df.corruption)]
    clean = df[df.corruption == "clean"].set_index("method")
    fig, axes = plt.subplots(2, len(kinds), figsize=(4 * len(kinds), 7), squeeze=False)
    for j, k in enumerate(kinds):
        d = df[df.corruption == k]
        for method in d.method.unique():
            dm = d[d.method == method].sort_values("severity")
            xs = [0, *dm.severity]
            axes[0, j].plot(xs, [clean.loc[method, "macro_f1_val"], *dm.macro_f1_val], "o-", label=method)
            axes[1, j].plot(xs, [clean.loc[method, "ece_val"], *dm.ece_val], "o-", label=method)
        axes[0, j].set_title(f"{k} (mức {LEVELS[k]})", fontsize=9)
        axes[1, j].set_xlabel("mức độ (0 = sạch)")
        for ax in axes[:, j]:
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("macro-F1 val")
    axes[1, 0].set_ylabel("ECE val")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle(f"Lệch phân phối tự tạo trên val: {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
