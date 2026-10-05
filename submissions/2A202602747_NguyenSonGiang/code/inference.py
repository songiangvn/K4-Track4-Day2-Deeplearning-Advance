"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val; nhiệt độ T khớp
trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện:
    predict_logits(model, loader, device, view=None)      -> (filenames, y_true, logits[N, 9])
    predict_views(model, loader, device, views_fn)        -> (filenames, y_true, [logits[N, 9]] * K)
    aggregate_views(list_of_logits, space)                -> probs[N, 9]
    fit_temperature(val_logits, val_labels)               -> float T
    apply_temperature(logits, T)                          -> probs
    ensemble_probs(list_of_probs)                         -> probs
    fuse_conv_bn(model)                                   -> model (BN đã gộp vào conv)

Quy ước view: TTA lật dùng ảnh val chuẩn (center-crop 224 từ ảnh 256). Multi-crop và multi-scale
dùng ẢNH GỐC 256x256 (loader không crop, xem full_image_transform) rồi tự cắt/resize trên GPU.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
# Chạy model
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def predict_views(model, loader, device, views_fn, amp: bool = True):
    """Chạy model trên mọi view do `views_fn(x) -> list[batch]` sinh ra từ mỗi batch.

    Trả về (filenames, y_true, list K mảng logit [N, 9] float32), đúng thứ tự của loader.
    """
    model.eval()
    names, ys, outs = [], [], None
    for x, y, f in loader:
        x = x.to(device, non_blocking=True)
        views = views_fn(x)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = [model(v).float().cpu() for v in views]
        if outs is None:
            outs = [[] for _ in logits]
        for k, lg in enumerate(logits):
            outs[k].append(lg)
        ys.append(y)
        names.extend(f)
    return names, torch.cat(ys).numpy(), [torch.cat(o).numpy() for o in outs]


def predict_logits(model, loader, device, view=None, amp: bool = True):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ view_hflip), hoặc None.
    """
    view = view or view_identity
    names, y, (logits,) = predict_views(model, loader, device, lambda x: [view(x)], amp)
    return names, y, logits


# --------------------------------------------------------------------------- #
# View
# --------------------------------------------------------------------------- #
def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W): đảo chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def view_vflip(x):
    """Lật dọc. Ảnh DeepWeeds chụp từ trên xuống nên lật dọc vẫn là ảnh hợp lệ."""
    return torch.flip(x, dims=[-2])


def views_hflip(x):
    """TTA K = 2: ảnh gốc và ảnh lật ngang."""
    return [x, view_hflip(x)]


def views_multicrop(x, crop: int = 224, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop` từ batch ảnh lớn hơn (ví dụ 256), và tuỳ chọn thêm
    bản lật ngang của từng crop (10 crop). Trả về list các batch."""
    h, w = x.shape[-2:]
    if crop > min(h, w):
        raise ValueError(f"crop {crop} lớn hơn ảnh {h}x{w}")
    t, l = (h - crop) // 2, (w - crop) // 2
    crops = [x[..., :crop, :crop], x[..., :crop, w - crop:], x[..., h - crop:, :crop],
             x[..., h - crop:, w - crop:], x[..., t:t + crop, l:l + crop]]
    if flip:
        crops += [view_hflip(c) for c in crops]
    return crops


def views_multiscale(x, sizes=(224, 256, 288)):
    """Resize (bilinear, antialias) cả batch về từng kích thước trong `sizes`, trả về list các batch.

    CNN có global pooling nhận được mọi kích thước. ViT cần tạo bằng dynamic_img_size=True
    (load_model(..., any_size=True)); Swin chỉ chạy ở một kích thước cố định nên không dùng được.
    """
    return [x if x.shape[-1] == s else
            F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=True)
            for s in sizes]


def load_model(run_path, device="cuda", img_size: int | None = None, any_size: bool = False):
    """Dựng lại model của một lần chạy (runs/<exp_id>/seed<k>/) từ config.json + best.pt.

    - img_size: kích thước đầu vào khác lúc train (chỉ cần cho Swin; CNN bỏ qua)
    - any_size: ViT/DeiT tạo với dynamic_img_size=True để nội suy pos-embed theo đầu vào
    Trả về (model.eval(), config dict). Báo lỗi nếu kiến trúc không chạy được ở kích thước đó.
    """
    import json
    from pathlib import Path

    import timm

    run_path = Path(run_path)
    cfg = json.loads((run_path / "config.json").read_text())
    kw = {}
    if any_size and ("vit" in cfg["backbone"] or "deit" in cfg["backbone"]):
        kw["dynamic_img_size"] = True
    if "swin" in cfg["backbone"]:
        kw["img_size"] = img_size if img_size is not None else cfg["img_size"]
    if "dinov2" in cfg["backbone"]:
        kw["img_size"] = cfg["img_size"]
    model = timm.create_model(cfg["backbone"], pretrained=False, num_classes=9, **kw)
    state = torch.load(run_path / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


# --------------------------------------------------------------------------- #
# Gộp, ensemble, hiệu chuẩn
# --------------------------------------------------------------------------- #
def softmax(logits, T: float = 1.0):
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Trả về xác suất (N, 9) đã chuẩn hoá.
    """
    stack = np.stack([np.asarray(l, dtype=np.float64) for l in logits_per_view])
    if space == "prob":
        return np.mean([softmax(l) for l in stack], axis=0)
    if space == "logit":
        return softmax(stack.mean(0))
    raise ValueError(f"space={space!r}; chọn 'prob' hoặc 'logit'")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file).

    Chi phí suy luận = số mô hình.
    """
    arr = np.stack([np.asarray(p, dtype=np.float64) for p in list_of_probs])
    if arr.ndim != 3:
        raise ValueError("Mọi mảng xác suất phải cùng dạng (N, K)")
    p = arr.mean(0)
    return p / p.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    Tìm lưới thô trên log T trong [-3, 3] rồi tinh bằng LBFGS trên log T (float64). Accuracy không
    đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    grid = torch.linspace(-3, 3, 121, dtype=torch.float64)
    nll = torch.stack([F.cross_entropy(z / t.exp(), y) for t in grid])
    log_t = grid[nll.argmin()].clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T)."""
    return softmax(logits, T)


def crossfit_temperature(val_logits, val_labels, n_splits: int = 2, seed: int = 0):
    """Ước lượng trung thực ECE sau temperature scaling khi chỉ có val: chia val thành `n_splits`
    phần (phân tầng theo lớp), khớp T trên các phần còn lại, áp lên phần giữ ra.

    Trả về (probs ngoài-mẫu [N, 9], list T của từng phần). T cuối cùng cho test vẫn khớp trên TOÀN BỘ val.
    """
    z, y = np.asarray(val_logits), np.asarray(val_labels)
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        fold[idx] = np.arange(len(idx)) % n_splits
    probs = np.empty((len(y), z.shape[1]))
    temps = []
    for k in range(n_splits):
        t = fit_temperature(z[fold != k], y[fold != k])
        probs[fold == k] = apply_temperature(z[fold == k], t)
        temps.append(t)
    return probs, temps


# --------------------------------------------------------------------------- #
# Gộp BatchNorm
# --------------------------------------------------------------------------- #
def _fused_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """w' = gamma * w / sqrt(var + eps); b' = beta + gamma * (b - mean) / sqrt(var + eps)."""
    fused = copy.deepcopy(conv)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps) if bn.affine else \
        1.0 / torch.sqrt(bn.running_var + bn.eps)
    shift = bn.bias if bn.affine else torch.zeros_like(bn.running_mean)
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused.weight = nn.Parameter((conv.weight * scale.reshape(-1, 1, 1, 1)).detach())
    fused.bias = nn.Parameter((shift + (b - bn.running_mean) * scale).detach())
    return fused


def _bn_replacement(bn: nn.BatchNorm2d) -> nn.Module:
    """BN thường -> Identity. BatchNormAct2d của timm (BN + dropout + activation trong một module)
    -> giữ lại phần dropout + activation."""
    if hasattr(bn, "act") and hasattr(bn, "drop"):
        return nn.Sequential(bn.drop, bn.act)
    return nn.Identity()


@torch.no_grad()
def fuse_conv_bn(model, example=None, atol: float = 1e-3):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75).

    Trả về bản sao đã gộp (model gốc giữ nguyên), kèm thuộc tính `fused_pairs` (số cặp đã gộp) và
    `fuse_max_abs_err` (sai số logit lớn nhất so với bản gốc trên `example`).

    Cách tìm cặp: trong mỗi module cha, một Conv2d mà module con ĐĂNG KÝ NGAY SAU nó là BatchNorm2d
    có cùng số kênh. Đúng với ResNet/ResNeXt/EfficientNet/MobileNetV3 của timm (thứ tự đăng ký =
    thứ tự chạy); vì là heuristic nên luôn kiểm tra bằng số: lệch quá `atol` thì báo lỗi.
    ViT, Swin, ConvNeXt dùng LayerNorm: không có cặp nào để gộp (fused_pairs = 0).
    """
    model = model.eval()
    fused_model = copy.deepcopy(model).eval()
    n = 0
    for parent in list(fused_model.modules()):
        children = list(parent.named_children())
        for (n1, m1), (n2, m2) in zip(children, children[1:]):
            if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d) \
                    and m1.out_channels == m2.num_features:
                setattr(parent, n1, _fused_conv(m1, m2))
                setattr(parent, n2, _bn_replacement(m2))
                n += 1
    fused_model.fused_pairs = n
    if example is not None:
        # Tắt TF32 khi kiểm tra: mặc định PyTorch cho cuDNN conv dùng TF32 (sai số ~1e-3), sẽ che
        # mất sai số thật của phép gộp.
        tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        try:
            err = float((model(example) - fused_model(example)).abs().max())
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
        fused_model.fuse_max_abs_err = err
        if err > atol:
            raise RuntimeError(f"Gộp BN làm lệch logit {err:.2e} > {atol}: có cặp conv/BN ghép sai")
    return fused_model
