"""final_predict.py - Bước 4: áp phương pháp suy luận đã chốt (trên val) lên TEST, đúng một lần mỗi seed.

    python final_predict.py --run F01/seed0 --method I01 --ts --exp-id F01
    python final_predict.py --run T00/seed0 --method I00 --exp-id T00          # mốc T00 + I00

Đây là NƠI DUY NHẤT trong code đánh giá trên test. train.py chạy với save_test_predictions=False.
Mỗi (exp_id, seed) chỉ được chạy test một lần: có file khoá runs/final/<exp_id>_seed<k>.json thì dừng.

`--method` (đã chọn ở Bước 3, xem run_inference.py):
    I00 | I01 | I01L | I02a | I02aL | I02b | I02bL | I02c | I02cL | I04_c<r> | I04_f<r>
    Phương pháp gộp xác suất được biểu diễn bằng log(xác suất trung bình) để temperature scaling
    áp dụng chung: softmax(log p / T).
`--ensemble R1 R2 ...`: thêm các model khác (mỗi model cùng `--method`), trung bình xác suất.
`--ts`: temperature scaling, T khớp trên VAL của chính pipeline này rồi áp sang test.

Ghi ra (pred_dir mặc định là predictions/ của bài nộp, định dạng eval.save_predictions):
    <exp_id>_seed<k>_test.csv         xác suất cuối (sau TS nếu --ts)
    <exp_id>_seed<k>_val.csv          xác suất val của cùng pipeline (cho eval.py grade --final-val)
    <exp_id>uncal_seed<k>_test.csv    bản chưa TS (chỉ khi --ts; cho eval.py grade --uncal)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

import dataset as D
import inference as I
import run_inference as RI
from eval import save_predictions

FINAL = RI.RUNS / "final"


def method_logits(ctx: RI.Ctx, method: str, split: str):
    """(names, y, z) với z là 'logit' của pipeline: logit thật, TB logit, hoặc log(TB xác suất)."""
    m = re.fullmatch(r"I04_([cf])(\d+)", method)
    if m:
        crop_pct = D.CROP_PCT if m.group(1) == "c" else 1.0
        return RI.resolution_logits(ctx, split, int(m.group(2)), crop_pct)
    m = re.fullmatch(r"(I0[0-2][abc]?)(L?)", method)
    if not m:
        raise ValueError(f"method {method!r} không hỗ trợ")
    names, y, views = RI.view_logits(ctx, m.group(1), split)
    if len(views) == 1:
        return names, y, views[0]
    if m.group(2) == "L":
        return names, y, np.mean(views, axis=0)
    return names, y, np.log(np.clip(I.aggregate_views(views, "prob"), 1e-12, None))


def pipeline(ctxs, method, split):
    outs = [method_logits(c, method, split) for c in ctxs]
    names, y, z0 = outs[0]
    for n, yy, _ in outs[1:]:
        assert list(n) == list(names) and np.array_equal(yy, y)
    if len(outs) == 1:
        return names, y, z0
    p = I.ensemble_probs([I.softmax(z) for _, _, z in outs])
    return names, y, np.log(np.clip(p, 1e-12, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--ensemble", nargs="*", default=[])
    ap.add_argument("--ts", action="store_true")
    ap.add_argument("--pred-dir", default=str(RI.SUB / "predictions"))
    ap.add_argument("--num-workers", type=int, default=10)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = RI.Ctx(RI.RUNS / args.run, dev, args.num_workers)
    seed = ctx.cfg["seed"]
    lock = FINAL / f"{args.exp_id}_seed{seed}.json"
    if lock.exists():
        raise SystemExit(f"{lock} đã tồn tại: test của {args.exp_id} seed{seed} đã chạy rồi (chỉ một lần).")
    ctxs = [ctx] + [RI.Ctx(RI.RUNS / r, dev, args.num_workers) for r in args.ensemble]

    # 1. val: khớp T (mọi quyết định xong trước khi đụng tới test)
    names_v, yv, zv = pipeline(ctxs, args.method, "val")
    T = I.fit_temperature(zv, yv) if args.ts else 1.0
    pv = I.apply_temperature(zv, T)
    pred = Path(args.pred_dir)
    save_predictions(pred / f"{args.exp_id}_seed{seed}_val.csv", names_v, yv, pv)

    # 2. test: đúng một lần
    FINAL.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"status": "started", "time": time.ctime()}))
    names_t, yt, zt = pipeline(ctxs, args.method, "test")
    save_predictions(pred / f"{args.exp_id}_seed{seed}_test.csv", names_t, yt, I.apply_temperature(zt, T))
    if args.ts:
        save_predictions(pred / f"{args.exp_id}uncal_seed{seed}_test.csv", names_t, yt, I.softmax(zt))
    np.savez_compressed(FINAL / f"{args.exp_id}_seed{seed}_logits.npz", val=zv, test=zt,
                        names_val=np.array(names_v), names_test=np.array(names_t))
    lock.write_text(json.dumps({
        "exp_id": args.exp_id, "seed": seed, "run": args.run, "ensemble": args.ensemble,
        "method": args.method, "temperature_scaling": args.ts, "T": T, "time": time.ctime(),
        "backbone": ctx.cfg["backbone"]}, indent=2))
    print(f"{args.exp_id} seed{seed}: method={args.method} ts={args.ts} T={T:.4f} -> {pred}")
    print("Chỉ số test: chạy eval.py score/grade (không in ở đây).")


if __name__ == "__main__":
    main()
