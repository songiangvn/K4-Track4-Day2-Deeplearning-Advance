"""gradcam.py - Grad-CAM để giải thích lỗi (điểm thưởng +1; Selvaraju et al. 2017).

    python gradcam.py --run F01/seed0 --split val --pairs "Chinee Apple:Snake Weed" "Snake Weed:Chinee Apple"
    python gradcam.py --run F01/seed0 --split test --n 8        # phân tích lỗi SAU chung kết

Grad-CAM tính trên đầu ra của model.forward_features (bản đồ đặc trưng cuối, trước pooling):
    w_c = trung bình không gian của d y_c / d A;   CAM = ReLU(sum_c w_c * A_c), phóng về 224.
Hỗ trợ đầu ra NCHW (ResNet, EfficientNet, MobileNet, ConvNeXt), NHWC (Swin) và NLC (ViT/DeiT:
bỏ token cls/prefix rồi xếp lại thành lưới patch).
Mỗi ô: ảnh + CAM của lớp DỰ ĐOÁN (sai) và CAM của lớp ĐÚNG, để thấy model nhìn vào đâu.
Chạy trên test chỉ để phân tích lỗi sau khi đã chốt kết quả (không dùng để chọn gì).
"""
from __future__ import annotations

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import dataset as D
import inference as I
import run_inference as RI


def feature_map(model, feats: torch.Tensor) -> torch.Tensor:
    """Đưa đầu ra forward_features về dạng (N, C, H, W)."""
    if feats.ndim == 4:
        name = type(model).__name__
        return feats.permute(0, 3, 1, 2) if name.startswith("SwinTransformer") else feats
    if feats.ndim == 3:                                   # ViT: (N, prefix + H*W, C)
        n_prefix = getattr(model, "num_prefix_tokens", 1)
        tok = feats[:, n_prefix:]
        side = int(math.isqrt(tok.shape[1]))
        return tok.transpose(1, 2).reshape(tok.shape[0], -1, side, side)
    raise ValueError(f"Không hỗ trợ đầu ra forward_features dạng {tuple(feats.shape)}")


def grad_cam(model, x: torch.Tensor, classes: list[int]) -> tuple[np.ndarray, torch.Tensor]:
    """CAM (N, H_in, W_in) trong [0, 1] cho lớp classes[i] của ảnh i, và logits."""
    model.eval()
    model.zero_grad(set_to_none=True)
    store = {}
    hook = None
    if type(model).__name__ == "VisionTransformer":
        # ViT phân loại bằng token cls: gradient về token patch ở ĐẦU RA cuối bằng 0. Lấy activation ở
        # đầu vào attention của block cuối (đầu ra norm1), nơi token patch còn ảnh hưởng tới cls.
        def keep(_, __, out):
            out.retain_grad()
            store["A"] = out
        hook = model.blocks[-1].norm1.register_forward_hook(keep)
    try:
        feats = model.forward_features(x)
        if hook is None:
            feats.retain_grad()
            store["A"] = feats
        logits = model.forward_head(feats)
        sel = logits[torch.arange(len(x)), torch.as_tensor(classes, device=x.device)]
        sel.sum().backward()
    finally:
        if hook is not None:
            hook.remove()
    A, G = feature_map(model, store["A"]), feature_map(model, store["A"].grad)
    w = G.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * A).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return cam.detach().cpu().numpy(), logits.detach()


def denorm(x):
    m = torch.tensor(D.IMAGENET_MEAN)[:, None, None]
    s = torch.tensor(D.IMAGENET_STD)[:, None, None]
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--pairs", nargs="*", default=["Chinee Apple:Snake Weed", "Snake Weed:Chinee Apple"],
                    help="cặp 'lớp thật:lớp dự đoán' cần xem; rỗng = mọi lỗi")
    ap.add_argument("--n", type=int, default=6, help="số ảnh mỗi cặp")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = RI.Ctx(RI.RUNS / args.run, dev, 8)
    model = I.load_model(ctx.run, dev)[0].float()
    names, y, logits = I.predict_logits(model, ctx.loader(args.split), dev)
    pred = logits.argmax(1)
    cls = {n: i for i, n in enumerate(D.CLASS_NAMES)}
    pairs = [tuple(cls[c] for c in p.split(":")) for p in args.pairs] if args.pairs else \
        sorted({(int(a), int(b)) for a, b in zip(y, pred) if a != b})
    ds = ctx.loader(args.split).dataset
    tag = args.tag or f"gradcam_{args.run.replace('/', '_')}_{args.split}"

    for t, p in pairs:
        idx = np.flatnonzero((y == t) & (pred == p))[: args.n]
        if len(idx) == 0:
            print(f"Không có ảnh {D.CLASS_NAMES[t]} -> {D.CLASS_NAMES[p]}")
            continue
        x = torch.stack([ds[i][0] for i in idx]).to(dev)
        cam_pred, lg = grad_cam(model, x, [p] * len(idx))
        cam_true, _ = grad_cam(model, x, [t] * len(idx))
        prob = lg.float().softmax(-1).cpu().numpy()
        fig, axes = plt.subplots(3, len(idx), figsize=(2.6 * len(idx), 8), squeeze=False)
        for j, i in enumerate(idx):
            img = denorm(x[j])
            axes[0, j].imshow(img)
            axes[0, j].set_title(f"{names[i]}\np({D.CLASS_NAMES[p][:10]})={prob[j, p]:.2f}", fontsize=7)
            for r, (cam, c) in enumerate(((cam_pred, p), (cam_true, t)), 1):
                axes[r, j].imshow(img)
                axes[r, j].imshow(cam[j], cmap="jet", alpha=0.45)
                axes[r, j].set_title(f"CAM lớp {D.CLASS_NAMES[c]}" + (" (dự đoán)" if r == 1 else " (đúng)"),
                                     fontsize=7)
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"{args.split}: thật {D.CLASS_NAMES[t]} → đoán {D.CLASS_NAMES[p]} "
                     f"({int(((y == t) & (pred == p)).sum())} ảnh) | {ctx.cfg['backbone']}", fontsize=10)
        fig.tight_layout()
        out = RI.SUB / "figures" / f"{tag}_{D.CLASS_NAMES[t].replace(' ', '')}_to_{D.CLASS_NAMES[p].replace(' ', '')}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print("Đã ghi", out)


if __name__ == "__main__":
    main()
