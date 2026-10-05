"""make_figures.py - biểu đồ tổng hợp cho báo cáo (đọc số từ runs/ và eval/, không tính lại gì).

    python make_figures.py

Ghi vào figures/:
  fig_backbones.png           macro-F1 val (mean ± std, 3 seed) của 7 backbone, công thức nền gốc vs mới
  fig_backbone_tradeoff.png   macro-F1 val theo GMAC (công thức nền mới)
  fig_ablation.png            Δ macro-F1 val so với T00 của từng thí nghiệm Bước 2 (Swin-T, MobileNetV3-L)
  fig_final_test.png          macro-F1 TEST của chung kết và mốc (mean ± std, 3 seed)
  fig_confusion_F01.png       ma trận nhầm lẫn test của F01 (cộng 3 seed, chuẩn hoá theo hàng)
Bảng màu: hai màu phân loại (#2a78d6 xanh, #eb6834 cam) đã kiểm tra bằng validator (đạt CVD); ma trận
nhầm lẫn dùng thang tuần tự một tông xanh.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import make_results as MR
from run_inference import SUB

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FIG = SUB / "figures"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "legend.frameon": False,
})

NAMES = {"resnet50": "ResNet-50", "resnext50_32x4d": "ResNeXt-50", "convnext_tiny.fb_in1k": "ConvNeXt-T",
         "deit_small_patch16_224": "DeiT-S", "swin_tiny_patch4_window7_224": "Swin-T",
         "efficientnet_b0": "EfficientNet-B0", "mobilenetv3_large_100": "MobileNetV3-L"}


def fig_backbones(a: pd.DataFrame):
    old = a[a.exp_id.str.fullmatch(r"B0\d")].set_index("backbone")
    new = a[a.exp_id.str.fullmatch(r"B1\d")].set_index("backbone")
    order = new.sort_values("macro_f1_val_mean").index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for d, color, off, lab in ((old, BLUE, -0.12, "Nền gốc B01–B07: val = CenterCrop 224 (không nội suy); "
                                                     "B02/B04/B05/B06 chỉ 1 seed (không có thanh std)"),
                               (new, ORANGE, 0.12, "Nền mới B11–B17: val = resize cả ảnh 256→224")):
        d = d.loc[order]
        ax.errorbar(d.macro_f1_val_mean, y + off, xerr=d.macro_f1_val_std, fmt="o", ms=7, color=color,
                    ecolor=color, elinewidth=1.5, capsize=3, label=lab, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
    for i, bb in enumerate(order):
        ax.plot([old.loc[bb, "macro_f1_val_mean"], new.loc[bb, "macro_f1_val_mean"]], [i - 0.12, i + 0.12],
                color=GRID, lw=1.5, zorder=1)
        ax.text(new.loc[bb, "macro_f1_val_mean"] + 0.006, i + 0.12, f"{new.loc[bb, 'macro_f1_val_mean']:.3f}",
                va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [NAMES[b] for b in order])
    ax.set_xlabel("macro-F1 val (mean ± std, 3 seed)")
    ax.set_title("Bước 1: backbone với cùng công thức nền, trước và sau khi sửa tiền xử lý val", fontsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=8)
    ax.set_title("Bước 1: backbone với cùng công thức nền, trước và sau khi sửa tiền xử lý val", fontsize=11,
                 pad=40)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_backbones.png", dpi=150)
    plt.close(fig)


def fig_backbone_tradeoff(a: pd.DataFrame):
    new = a[a.exp_id.str.fullmatch(r"B1\d")]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(new.gmacs, new.macro_f1_val_mean, yerr=new.macro_f1_val_std, fmt="o", ms=8, color=BLUE,
                ecolor=BLUE, elinewidth=1.5, capsize=3, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for _, r in new.iterrows():
        ax.annotate(f"{NAMES[r.backbone]} ({r.exp_id})", (r.gmacs, r.macro_f1_val_mean), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("GMAC / ảnh 224×224 (thang log)")
    ax.set_ylabel("macro-F1 val")
    ax.set_title("Backbone (công thức nền mới): chất lượng theo chi phí tính toán", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_backbone_tradeoff.png", dpi=150)
    plt.close(fig)


def fig_ablation(t: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    for ax, suf, title, lim in ((axes[0], "s", "Swin-T (mốc T00s)", 0.012),
                                (axes[1], "m", "MobileNetV3-L (mốc T00m)", 0.042)):
        d = t[t.exp_id.str.fullmatch(rf"[TC]\d\d{suf}") & ~t.exp_id.str.fullmatch(r"T0[12][rcm]")]
        d = d.sort_values("delta_macro_f1_vs_baseline")
        y = np.arange(len(d))
        x = d.delta_macro_f1_vs_baseline.clip(-lim, lim)
        ax.barh(y, x, height=0.7, color=[ORANGE if e.startswith("C") else BLUE for e in d.exp_id],
                edgecolor=SURFACE, linewidth=2)
        ax.errorbar(x, y, xerr=d.max_std, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        for i, (_, r) in enumerate(d.iterrows()):
            if abs(r.delta_macro_f1_vs_baseline) > lim:
                ax.text(-lim + lim * 0.03, i, f"{r.delta_macro_f1_vs_baseline:+.3f} ◂", va="center", fontsize=7,
                        color=SURFACE, fontweight="bold")
        labels = [f"{r.exp_id} {r.diff_vs_baseline[:38]}{' *' if r.verdict.startswith('tốt') else ''}"
                  for _, r in d.iterrows()]
        ax.set_yticks(y, labels, fontsize=7)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("Δ macro-F1 val so với mốc (mean, 3 seed)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, label="một yếu tố (T)"), Patch(color=ORANGE, label="kết hợp (C)")],
               loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Bước 2: đóng góp của từng yếu tố công thức huấn luyện\n"
                 "* = Δ > std; thanh lỗi = std lớn hơn của hai nhóm; thanh vượt trục bị cắt, giá trị ghi trên thanh",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(FIG / "fig_ablation.png", dpi=150)
    plt.close(fig)


def fig_final():
    rows = []
    for eid, lab, color in (("T00", "T00 Swin-T mốc + I00", BLUE), ("F01", "F01 Swin-T C01s + 10 crop + TS", BLUE),
                            ("T00m", "T00m MobileNetV3 mốc + I00", ORANGE),
                            ("F02", "F02 MobileNetV3 C03m + 5 crop + TS", ORANGE),
                            ("F03", "F03 MobileNetV3 + KD + 1 view 352 + TS", ORANGE)):
        s = json.loads((SUB / "eval" / f"{eid}_summary.json").read_text())
        rows.append((lab, s["macro_f1"]["mean"], s["macro_f1"]["std"], s["top1"]["mean"], color))
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, m, sd, top1, color) in zip(y, rows):
        ax.errorbar(m, yi, xerr=sd, fmt="o", ms=8, color=color, ecolor=color, elinewidth=1.5, capsize=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.text(m + 0.003, yi, f"{m:.4f} ± {sd:.4f}  (top-1 {top1 * 100:.2f}%)", va="center", fontsize=8,
                color=INK2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0.915, 1.012)
    ax.set_xlabel("macro-F1 TEST fold 0 (mean ± std, 3 seed; tính bằng eval.py)")
    ax.set_title("Chung kết so với mốc của từng backbone (xanh: Swin-T · cam: MobileNetV3-L)", fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_final_test.png", dpi=150)
    plt.close(fig)


def fig_confusion(eid: str = "F01"):
    cm = pd.read_csv(SUB / "eval" / f"{eid}_confusion_sum.csv", index_col=0)
    names = [c.replace("true_", "") for c in cm.index]
    m = cm.to_numpy(dtype=float)
    pct = m / m.sum(1, keepdims=True) * 100
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(np.sqrt(pct), cmap=cmap, vmin=0, vmax=10)        # căn bậc hai để ô nhầm lẫn nhỏ vẫn thấy
    for i in range(len(names)):
        for j in range(len(names)):
            if m[i, j] == 0:
                continue
            dark = np.sqrt(pct[i, j]) > 5.5
            ax.text(j, i, f"{pct[i, j]:.1f}%\n({int(m[i, j])})", ha="center", va="center", fontsize=7,
                    color="#ffffff" if dark else INK)
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("lớp dự đoán")
    ax.set_ylabel("lớp thật")
    ax.grid(False)
    ax.set_title(f"{eid}: ma trận nhầm lẫn TEST (cộng 3 seed; % theo hàng, số ảnh trong ngoặc)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"fig_confusion_{eid}.png", dpi=150)
    plt.close(fig)


# Nhóm phương pháp suy luận: (màu, marker). 5 màu đầu của bảng phân loại, đã chạy validator; 3 màu có
# tương phản < 3:1 nên mọi điểm đều có nhãn trực tiếp và mỗi nhóm có marker riêng (mã hoá phụ).
GROUPS = {"mốc": ("#2a78d6", "o"), "TTA (lật / crop / tỉ lệ)": ("#eb6834", "s"),
          "độ phân giải test": ("#1baf7a", "^"), "ensemble": ("#eda100", "D"),
          "triển khai (gộp BN / FP16 / AMP)": ("#e87ba4", "v")}


def _tradeoff_points(d: pd.DataFrame, S: int) -> pd.DataFrame:
    """Mỗi phương pháp một điểm: bỏ bản trùng chi phí và các cấu hình KHÔNG nội suy (lỗi đã biết)."""
    keep = {"I00": ("mốc", f"I00 1 view ({S})"), "I01L": ("TTA (lật / crop / tỉ lệ)", "I01 lật, K=2"),
            "I02aL": ("TTA (lật / crop / tỉ lệ)", "I02a 5 crop"), "I02bL": ("TTA (lật / crop / tỉ lệ)", "I02b 10 crop"),
            "I02cL": ("TTA (lật / crop / tỉ lệ)", "I02c 3 tỉ lệ"), "I05": ("ensemble", "I05 ensemble 3 seed"),
            "I08a": ("triển khai (gộp BN / FP16 / AMP)", "I08a gộp BN"),
            "I08b": ("triển khai (gộp BN / FP16 / AMP)", "I08b FP16"),
            "I08c": ("triển khai (gộp BN / FP16 / AMP)", "I08c AMP"),
            "I08d": ("triển khai (gộp BN / FP16 / AMP)", "I08d gộp BN + FP16")}
    for e in d.exp_id:
        if e.startswith("I04_f"):
            r = int(e[5:])
            if r > S and r != 256:                   # 256 = ảnh gốc không nội suy; r = S trùng I00
                keep[e] = ("độ phân giải test", f"I04 test ở {r}")
    rows = d[d.exp_id.isin(keep)].dropna(subset=["p50_ms_b1", "macro_f1_val"]).copy()
    if "note" in rows:
        rows = rows[~rows.note.fillna("").str.startswith("không có BN")]   # gộp BN không áp dụng cho mạng LN
    rows["group"] = rows.exp_id.map(lambda e: keep[e][0])
    rows["label"] = rows.exp_id.map(lambda e: keep[e][1])
    return rows


def _repel(ax, xs, ys, labels, dx_px=7, pad_px=2, iters=300):
    """Nhãn bên phải mỗi điểm; dùng khung bao THẬT của chữ, đẩy theo chiều dọc các cặp nhãn chồng nhau
    (và nhãn đè lên điểm khác) cho tới khi hết chồng; vẽ đường dẫn khi nhãn bị dời xa điểm."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.c_[xs, ys])
    texts = [ax.text(0, 0, lab, fontsize=7.5, color=INK2, va="center", ha="left") for lab in labels]
    w = np.array([t.get_window_extent(rend).width for t in texts]) + 2 * pad_px
    h = np.array([t.get_window_extent(rend).height for t in texts]) + 2 * pad_px
    x0 = pts[:, 0] + dx_px                                   # cạnh trái nhãn (px)
    y = pts[:, 1].astype(float).copy()                       # tâm nhãn (px)
    for _ in range(iters):
        moved = False
        for a in range(len(y)):
            for b in range(a + 1, len(y)):
                if x0[a] < x0[b] + w[b] and x0[b] < x0[a] + w[a]:            # chồng theo chiều ngang
                    gap = (h[a] + h[b]) / 2 - abs(y[a] - y[b])
                    if gap > 0:
                        sgn = 1 if (y[a], a) > (y[b], b) else -1
                        y[a] += sgn * gap / 2 + sgn * 0.1
                        y[b] -= sgn * gap / 2 + sgn * 0.1
                        moved = True
            for k in range(len(y)):                                          # nhãn a đè lên điểm k
                if k != a and x0[a] - dx_px < pts[k, 0] < x0[a] + w[a] and abs(y[a] - pts[k, 1]) < h[a] / 2 + 4:
                    y[a] += (1 if y[a] >= pts[k, 1] else -1) * 1.5
                    moved = True
        if not moved:
            break
    inv = ax.transData.inverted()
    for k, t in enumerate(texts):
        t.remove()
        tx, ty = inv.transform((x0[k], y[k]))
        far = abs(y[k] - pts[k, 1]) > 3
        ax.annotate(labels[k], (xs[k], ys[k]), xytext=(tx, ty), textcoords="data", fontsize=7.5, color=INK2,
                    va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color="#b9b8b2", lw=0.8, shrinkA=0, shrinkB=4) if far else None)


def fig_tradeoff(tag: str) -> None:
    """Đánh đổi độ chính xác và độ trễ của một bảng run_inference (runs/inference/<tag>/)."""
    from matplotlib.ticker import FuncFormatter, LogLocator
    from run_inference import RUNS
    d = pd.read_csv(RUNS / "inference" / tag / "inference_val.csv")
    model = d["model"].dropna().iloc[0]
    run = model.split(" ")[0]
    S = json.loads((RUNS / run / "config.json").read_text())["img_size"]
    pts = _tradeoff_points(d, S)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for g, (color, marker) in GROUPS.items():
        q = pts[pts.group == g]
        if len(q):
            ax.scatter(q.p50_ms_b1, q.macro_f1_val, s=60, color=color, marker=marker, label=g, zorder=3,
                       edgecolors=SURFACE, linewidths=1.5)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 1.5, 2, 3, 5, 7)))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=()))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    x0, x1 = pts.p50_ms_b1.min(), pts.p50_ms_b1.max()
    ax.set_xlim(x0 / 1.1, x1 * 1.6)                      # chừa chỗ bên phải cho nhãn
    y0, y1 = pts.macro_f1_val.min(), pts.macro_f1_val.max()
    pad = max(0.002, (y1 - y0) * 0.12)
    ax.set_ylim(y0 - pad, y1 + pad)
    _repel(ax, pts.p50_ms_b1.to_numpy(), pts.macro_f1_val.to_numpy(), pts.label.tolist())
    ax.set_xlabel("độ trễ p50 batch 1 (ms, thang log; chỉ forward mạng, FP32 thật trừ FP16/AMP)")
    ax.set_ylabel("macro-F1 val (1 model)")
    gpu = d["gpu"].dropna().iloc[0]
    ax.set_title(f"Đánh đổi độ chính xác và độ trễ: {model} · {gpu}\n"
                 "một điểm mỗi phương pháp (gộp logit) · đủ số liệu ở sheet Inference của results.xlsx",
                 fontsize=9.5)
    ax.legend(loc="lower right", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(FIG / f"{tag}_tradeoff.png", dpi=150)
    plt.close(fig)


def main():
    runs = MR.load_runs()
    a = MR.agg(runs)
    fig_backbones(a)
    fig_backbone_tradeoff(a)
    fig_ablation(MR.training_sheet(runs))
    fig_final()
    for eid in ("F01", "F03"):
        fig_confusion(eid)
    from run_inference import RUNS
    for d in sorted((RUNS / "inference").glob("I_*")):
        if (d / "inference_val.csv").exists():
            fig_tradeoff(d.name)
    print("Đã ghi", sorted(p.name for p in FIG.glob("fig_*.png")))


if __name__ == "__main__":
    main()
