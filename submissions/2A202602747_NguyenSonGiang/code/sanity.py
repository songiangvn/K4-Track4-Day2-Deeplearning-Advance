"""sanity.py - kiểm tra pipeline trước khi chạy thật (GUIDE.md mục 1.3, slide trang 59).

    python sanity.py          # từ thư mục code/, cần GPU (~1-2 phút)

  1. loss ban đầu (head mới, chưa train) trên 1 batch val của mọi backbone ≈ ln 9 = 2.197
  2. overfit 1 batch nhỏ (16 ảnh train, không augmentation) tới loss gần 0
  3. ảnh sau augmentation (đã giải chuẩn hoá) kèm nhãn, cho mọi mức `aug`, và một batch CutMix/Mixup
  4. model.eval() cho kết quả giống nhau giữa batch 1 ảnh và batch nhiều ảnh (BN dùng running stats)
Kết quả: ../figures/sanity_*.png và ../sanity.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import dataset as D
from losses import mix_batch
from model import build_model, set_train_mode
from train import set_seed

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "data"
OUT = HERE.parent
BACKBONES = ["resnet50", "resnext50_32x4d", "convnext_tiny.fb_in1k", "deit_small_patch16_224",
             "swin_tiny_patch4_window7_224", "efficientnet_b0", "mobilenetv3_large_100"]


def denorm(x):
    m = torch.tensor(D.IMAGENET_MEAN)[:, None, None]
    s = torch.tensor(D.IMAGENET_STD)[:, None, None]
    return (x * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def grid(images, titles, path, ncol=6, suptitle=""):
    nrow = math.ceil(len(images) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.2, nrow * 2.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, im, t in zip(axes.flat, images, titles):
        ax.imshow(denorm(im))
        ax.set_title(t, fontsize=7)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    set_seed(0)
    dev = torch.device("cuda")
    (OUT / "figures").mkdir(exist_ok=True)
    train_df, val_df, _ = D.load_split(DATA / "labels")
    store = D.load_byte_store(DATA / "images")
    res = {"ln9": math.log(9)}

    # 1. loss ban đầu
    val_ds = D.DeepWeedsDataset(val_df, DATA / "images", D.build_transforms(False), cache=store)
    xb, yb, _ = next(iter(D.make_loader(None, None, None, 64, False, num_workers=4, dataset=val_ds)))
    xb, yb = xb.to(dev), yb.to(dev)
    res["initial_loss"] = {}
    for name in BACKBONES:
        m = build_model(name).to(dev).eval()
        with torch.inference_mode():
            res["initial_loss"][name] = float(F.cross_entropy(m(xb), yb))
        print(f"loss ban đầu {name:32s} {res['initial_loss'][name]:.4f}  (ln 9 = {math.log(9):.4f})")

    # 4. eval(): batch 1 ảnh và batch 64 ảnh cho cùng logit
    m = build_model("resnet50").to(dev).eval()
    with torch.inference_mode():
        a = m(xb)[:1]
        b = m(xb[:1])
    res["eval_batch_invariance_maxabs"] = float((a - b).abs().max())
    print("eval(): |logit(batch 64)[0] - logit(batch 1)| max =", res["eval_batch_invariance_maxabs"])

    # 2. overfit 1 batch nhỏ
    tr_small = train_df.groupby("Label").head(2).head(16)
    ds = D.DeepWeedsDataset(tr_small, DATA / "images", D.build_transforms(False), cache=store)
    x16 = torch.stack([ds[i][0] for i in range(len(ds))]).to(dev)
    y16 = torch.tensor([ds[i][1] for i in range(len(ds))], device=dev)
    m = build_model("resnet50").to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    losses = []
    for step in range(150):
        set_train_mode(m)
        loss = F.cross_entropy(m(x16), y16)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    m.eval()
    with torch.inference_mode():
        acc = float((m(x16).argmax(1) == y16).float().mean())
    res["overfit"] = {"n_images": len(ds), "steps": 150, "loss_first": losses[0], "loss_last": losses[-1],
                      "acc_eval_mode": acc}
    print(f"overfit 16 ảnh: loss {losses[0]:.4f} -> {losses[-1]:.5f}, acc (eval) {acc:.3f}")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.semilogy(losses)
    ax.axhline(math.log(9), ls=":", c="gray", label="ln 9")
    ax.set_xlabel("bước"), ax.set_ylabel("CE (log)"), ax.set_title("Overfit 1 batch 16 ảnh (resnet50)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "sanity_overfit_batch.png", dpi=120)
    plt.close(fig)

    # 3. ảnh sau augmentation + nhãn
    sample = train_df.groupby("Label").head(1)
    for aug in D.AUGS:
        ds = D.DeepWeedsDataset(sample, DATA / "images", D.build_transforms(True, 224, aug), cache=store)
        ims, titles = [], []
        for i in range(len(ds)):
            for _ in range(2):
                x, y, f = ds[i]
                ims.append(x), titles.append(f"{D.CLASS_NAMES[y]}\n{f}")
        grid(ims, titles, OUT / "figures" / f"sanity_aug_{aug}.png", ncol=6,
             suptitle=f"aug={aug}: ảnh sau augmentation (đã giải chuẩn hoá), 2 lần mỗi ảnh")

    ds = D.DeepWeedsDataset(sample, DATA / "images", D.build_transforms(True), cache=store)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor([ds[i][1] for i in range(len(ds))])
    for mode in ("cutmix", "mixup"):
        xm, (ya, yb_, lam) = mix_batch(x, y, 1.0, mode, rng=np.random.default_rng(3))
        titles = [f"{lam:.2f}·{D.CLASS_NAMES[a][:12]}\n+{1 - lam:.2f}·{D.CLASS_NAMES[b][:12]}"
                  for a, b in zip(ya.tolist(), yb_.tolist())]
        grid(list(xm), titles, OUT / "figures" / f"sanity_{mode}.png", ncol=5,
             suptitle=f"{mode}: lam = {lam:.3f} (trộn cả ảnh lẫn nhãn)")

    (OUT / "sanity.json").write_text(json.dumps(res, indent=2))
    print("Đã ghi", OUT / "sanity.json")


if __name__ == "__main__":
    main()
