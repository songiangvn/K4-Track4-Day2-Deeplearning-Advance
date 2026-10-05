"""res_sweep.py - macro-F1 VAL của các lần chạy theo độ phân giải kiểm tra (chẩn đoán lệch tỉ lệ
train/test, FixRes; slide trang 68). Chỉ dùng val.

    python res_sweep.py                       # mọi runs/B*/seed*
    python res_sweep.py --pattern "T0*/seed*"

Ghi/ghép vào runs/res_sweep.csv (exp_id, seed, backbone, test_res, macro_f1_val, top1_val).
Tiền xử lý: giống val của lần chạy (crop_pct trong config), ở độ phân giải r.
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch

import dataset as D
import inference as I
import run_inference as RI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="B*/seed*")
    ap.add_argument("--resolutions", nargs="*", type=int, default=[224, 256, 288, 320, 384])
    args = ap.parse_args()
    out = RI.RUNS / "res_sweep.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["exp_id", "seed", "test_res"])
    done = set(zip(old.exp_id, old.seed.astype(int), old.test_res.astype(int)))
    dev = torch.device("cuda")
    rows = []
    for run in sorted(RI.RUNS.glob(args.pattern)):
        if not (run / "best.pt").exists():
            continue
        ctx = RI.Ctx(run, dev)
        eid, seed = ctx.cfg["exp_id"], ctx.cfg["seed"]
        for r in args.resolutions:
            if (eid, seed, r) in done:
                continue
            _, y, lg = RI.resolution_logits(ctx, "val", r, ctx.crop_pct)
            m = RI.metrics(y, I.softmax(lg))
            rows.append({"exp_id": eid, "seed": seed, "backbone": ctx.cfg["backbone"], "test_res": r,
                         "macro_f1_val": m["macro_f1_val"], "top1_val": m["top1_val"], "ece_val": m["ece_val"]})
            print(f"{eid} seed{seed} @{r}: F1={m['macro_f1_val']:.4f}", flush=True)
        ctx.models.clear()
        torch.cuda.empty_cache()
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(out, index=False)
    print("Đã ghi", out)


if __name__ == "__main__":
    main()
