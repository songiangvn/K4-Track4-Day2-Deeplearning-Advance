"""method_select.py - chọn phương pháp suy luận cho chung kết trên VAL, qua nhiều seed (GUIDE N4).

    python method_select.py --runs C01s/seed0 C01s/seed1 C01s/seed2 --tag MS_C01s

run_inference.py so mọi phương pháp trên MỘT model; chênh lệch giữa các phương pháp nhỏ (~0.002-0.009)
nên cần biết nó có vượt nhiễu giữa các seed không. Với mỗi lần chạy và mỗi phương pháp ứng viên, tính
macro-F1 / top-1 / ECE val (thêm ECE sau temperature scaling khớp chéo 2 phần val), rồi Δ macro-F1 THEO
CẶP so với I00 của cùng model: mean ± std (ddof=1) qua seed.
Ghi runs/inference/<tag>.csv (từng seed) và <tag>_summary.csv (tổng hợp). Không chạm test.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

import final_predict as FP
import inference as I
import run_inference as RI

DEFAULT_METHODS = ["I00", "I01L", "I02aL", "I02bL", "I04_f320", "I04_f352", "I04_c352"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")

    rows = []
    for run in args.runs:
        ctx = RI.Ctx(RI.RUNS / run, dev)
        for meth in args.methods:
            try:
                _, y, z = FP.method_logits(ctx, meth, "val")
            except RI.NotApplicable as e:
                print(run, meth, "bỏ qua:", e)
                continue
            m = RI.metrics(y, I.softmax(z))
            p_ts, _ = I.crossfit_temperature(z, y)
            rows.append({"run": run, "seed": ctx.cfg["seed"], "method": meth, **m,
                         "ece_val_ts_crossfit": RI.metrics(y, p_ts)["ece_val"]})
            print(f"{run} {meth:9s} F1={m['macro_f1_val']:.4f} ECE={m['ece_val']:.4f}", flush=True)
        ctx.models.clear()
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    base = df[df.method == "I00"].set_index("run")["macro_f1_val"]
    df["delta_vs_I00"] = df["macro_f1_val"] - df["run"].map(base)
    out = RI.RUNS / "inference"
    df.to_csv(out / f"{args.tag}.csv", index=False)
    g = df.groupby("method", sort=False)
    summ = pd.DataFrame({
        "n_seeds": g.size(),
        "macro_f1_val_mean": g["macro_f1_val"].mean(), "macro_f1_val_std": g["macro_f1_val"].std(ddof=1),
        "delta_vs_I00_mean": g["delta_vs_I00"].mean(), "delta_vs_I00_std": g["delta_vs_I00"].std(ddof=1),
        "ece_val_mean": g["ece_val"].mean(), "ece_val_ts_mean": g["ece_val_ts_crossfit"].mean(),
    }).reset_index()
    summ["verdict"] = np.where(summ.method == "I00", "mốc",
                       np.where(summ.delta_vs_I00_mean > summ.delta_vs_I00_std, "tốt hơn I00 (Δ > std)",
                       np.where(summ.delta_vs_I00_mean < -summ.delta_vs_I00_std, "kém hơn I00",
                                "không phân biệt được")))
    summ.to_csv(out / f"{args.tag}_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
