"""eda.py - Bước 0: kiểm tra chia dữ liệu (README mục 2.1) và EDA (GUIDE mục 1.2).

    python eda.py            # từ thư mục code/; ghi ra ../figures và ../eda_split_check.json

Sinh ra:
  figures/eda_class_distribution.png   phân bố lớp theo train/val/test, đối chiếu Table 1 của bài báo
  figures/eda_samples.png              4 ảnh ngẫu nhiên mỗi lớp (từ train)
  figures/eda_label_mismatch.png       ảnh có nhãn fold khác labels.csv
  eda_split_check.json                 số liệu để dán vào báo cáo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import dataset as D

HERE = Path(__file__).resolve().parent
# Table 1 của bài báo (Olsen et al., 2019), cùng thứ tự CLASS_NAMES
PAPER_COUNTS = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]


def plot_distribution(per_class: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(per_class))
    w = 0.27
    for j, split in enumerate(("train", "val", "test")):
        ax.bar(x + (j - 1) * w, per_class[split], w, label=split)
    ax.scatter(x, PAPER_COUNTS, marker="_", s=400, c="k", label="Tổng theo Table 1 bài báo")
    ax.scatter(x, per_class["total"], marker="x", c="crimson", label="Tổng đếm được")
    ax.set_yscale("log")
    ax.set_xticks(x, per_class.index, rotation=25, ha="right")
    ax.set_ylabel("Số ảnh (thang log)")
    ax.set_title("DeepWeeds fold 0: phân bố lớp theo tập")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_samples(df: pd.DataFrame, images_dir: Path, path: Path, per_class: int = 4, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(D.NUM_CLASSES, per_class, figsize=(per_class * 2.1, D.NUM_CLASSES * 2.1))
    for c in range(D.NUM_CLASSES):
        files = df.loc[df["Label"] == c, "Filename"].to_numpy()
        for j, f in enumerate(rng.choice(files, per_class, replace=False)):
            ax = axes[c, j]
            with Image.open(images_dir / f) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(D.CLASS_NAMES[c], fontsize=9)
    fig.suptitle("Ảnh mẫu (train, ngẫu nhiên seed=0)", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE.parents[2] / "data"))
    ap.add_argument("--out", default=str(HERE.parent))
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    images_dir, labels_dir = data / "images", data / "labels"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = D.load_split(labels_dir)
    rep = D.check_split(train_df, val_df, test_df, images_dir, labels_csv=labels_dir / "labels.csv")
    per_class = rep["per_class"]
    per_class["paper_table1"] = PAPER_COUNTS
    per_class["diff_vs_paper"] = per_class["total"] - per_class["paper_table1"]
    print(per_class.to_string())

    # thống kê ảnh: kích thước, chế độ màu, mean/std kênh trên một mẫu train
    rng = np.random.default_rng(0)
    sample = rng.choice(train_df["Filename"].to_numpy(), 500, replace=False)
    sizes, modes, pix = set(), set(), []
    for f in sample:
        with Image.open(images_dir / f) as im:
            sizes.add(im.size), modes.add(im.mode)
            pix.append(np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3) / 255)
    pix = np.concatenate(pix)
    stats = {"sizes": sorted(map(list, sizes)), "modes": sorted(modes),
             "mean_rgb": pix.mean(0).round(4).tolist(), "std_rgb": pix.std(0).round(4).tolist(),
             "n_sampled": len(sample)}
    print("Thống kê ảnh (500 ảnh train):", stats)

    plot_distribution(per_class, out / "figures" / "eda_class_distribution.png")
    plot_samples(train_df, images_dir, out / "figures" / "eda_samples.png")

    if rep["label_mismatch"]:
        m = rep["label_mismatch"]
        fig, axes = plt.subplots(1, len(m), figsize=(3.2 * len(m), 3.4), squeeze=False)
        for ax, r in zip(axes[0], m):
            with Image.open(images_dir / r["Filename"]) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{r['Filename']}\nfold: {D.CLASS_NAMES[r['Label']]} | "
                         f"labels.csv: {D.CLASS_NAMES[r['Label_ref']]}", fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "figures" / "eda_label_mismatch.png", dpi=120)
        plt.close(fig)

    result = {
        "n": rep["n"], "frac": rep["frac"], "union": rep["union"], "overlap": rep["overlap"],
        "missing_files": rep["missing_files"], "label_mismatch": rep["label_mismatch"],
        "imbalance_ratio": rep["imbalance_ratio"],
        "per_class": per_class.reset_index(names="class").to_dict("records"),
        "image_stats": stats,
    }
    (out / "eda_split_check.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("Đã ghi", out / "eda_split_check.json")


if __name__ == "__main__":
    main()
