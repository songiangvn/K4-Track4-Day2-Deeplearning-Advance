"""make_results.py - gom mọi kết quả thành results.xlsx (GUIDE mục 6.1).

    python make_results.py [--final F01 --baseline T00] [--realtime F02]

Nguồn (mọi số đều đọc từ file do lần chạy thật sinh ra, truy ngược được tới exp_id):
  runs/<exp_id>/seed<k>/summary.json, history.csv      -> Backbones, Training
  runs/res_sweep.csv (res_sweep.py)                     -> cột "F1 val ở độ phân giải test tốt nhất"
  runs/inference/<tag>/inference_val.csv, latency.csv   -> Inference, Latency
  predictions/<id>_seed<k>_{val,test}.csv               -> Final, PerClass (tính bằng eval.load_group)
Sheet chưa có dữ liệu vẫn được tạo với tiêu đề cột và một dòng ghi chú.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import run_exps as R
from run_inference import RUNS, SUB, LABELS
import eval as ev

HARD = {"Chinee Apple": 0, "Snake Weed": 7}
AXES = {"A": "A. khởi tạo", "B": "B. augmentation", "C": "C. loss", "D": "D. cân bằng mẫu",
        "E": "E. LR/optimizer", "F": "F. chính quy hoá", "G": "G. độ phân giải/thời gian"}


def load_runs() -> pd.DataFrame:
    rows = []
    for f in sorted(RUNS.glob("*/seed*/summary.json")):
        s = json.loads(f.read_text())
        cfg = json.loads((f.parent / "config.json").read_text())
        v = s["val"]
        rows.append({
            "exp_id": s["exp_id"], "seed": s["seed"], "backbone": s["backbone"], "weight_tag": s["weight_tag"],
            "params_M": s["params_M"], "gmacs": s["gmacs"], "img_size": cfg["img_size"], "epochs": cfg["epochs"],
            "best_epoch": s["best_epoch"], "macro_f1_val": v["macro_f1"], "top1_val": v["top1"],
            "balanced_acc_val": v["balanced_acc"], "ece_val": v["ece"],
            "f1_chinee_val": v["f1_per_class"][0], "f1_snake_val": v["f1_per_class"][7],
            "train_s_per_epoch": s["train_time_per_epoch_s"],
            "latency_b1_p50_ms": s.get("latency_b1_fp32", {}).get("p50_ms"),
            "gpu": s["env"]["gpu"], "desc": cfg.get("desc", ""), "cfg": cfg,
        })
    return pd.DataFrame(rows)


def agg(df: pd.DataFrame, keys=("exp_id",)) -> pd.DataFrame:
    num = ["macro_f1_val", "top1_val", "balanced_acc_val", "ece_val", "f1_chinee_val", "f1_snake_val",
           "train_s_per_epoch", "latency_b1_p50_ms", "best_epoch"]
    g = df.groupby(list(keys))
    out = g[num].mean().add_suffix("_mean")
    std = g[["macro_f1_val", "top1_val"]].std(ddof=1).add_suffix("_std")
    first = g[["backbone", "weight_tag", "params_M", "gmacs", "img_size", "epochs", "gpu", "desc"]].first()
    seeds = g["seed"].apply(lambda s: ",".join(map(str, sorted(s)))).rename("seeds")
    n = g.size().rename("n_seeds")
    return pd.concat([first, n, seeds, out, std], axis=1).reset_index()


def diff_vs(cfg: dict, base: dict) -> str:
    skip = {"exp_id", "desc", "seed", "out_dir", "pred_dir", "curves_dir", "images_dir", "labels_dir",
            "weight_tag", "params_M", "gmacs", "env", "train_counts", "class_weight", "num_workers"}
    # trường thêm vào Config sau một lần chạy cũ thì lấy giá trị mặc định
    defaults = {f.name: f.default for f in dataclasses.fields(R.Config)}
    keys = sorted((set(cfg) | set(base)) - skip)
    d = [f"{k}={cfg.get(k, defaults.get(k))!r}" for k in keys
         if cfg.get(k, defaults.get(k)) != base.get(k, defaults.get(k))]
    return "; ".join(d) or "(giống mốc)"


def baseline_of(exp_id: str) -> str | None:
    """Mốc so sánh của một thí nghiệm huấn luyện: S -> B tương ứng, T..<x> -> T00<x>."""
    m = re.fullmatch(r"T0[12]([rcm])", exp_id)
    if m:
        return R.SCALE_BACKBONES[m.group(1)]
    if re.fullmatch(r"X1\d", exp_id):
        return R.KD_STUDENT_BASE                       # KD: so với chính công thức của student
    m = re.fullmatch(r"[TC]\d\d([a-z])", exp_id)
    if m:
        return f"T00{m.group(1)}"
    return None


def training_sheet(runs: pd.DataFrame) -> pd.DataFrame:
    a = agg(runs).set_index("exp_id")
    first_cfg = runs.groupby("exp_id")["cfg"].first()
    rows = []
    for eid in a.index:
        base = baseline_of(eid)
        if base is None or base not in a.index or eid == base:
            continue
        r, b = a.loc[eid], a.loc[base]
        delta = r["macro_f1_val_mean"] - b["macro_f1_val_mean"]
        # std lớn hơn của hai nhóm; NaN nếu một bên chỉ có 1 seed (khi đó không kết luận)
        s = max(r["macro_f1_val_std"], b["macro_f1_val_std"])
        if r["n_seeds"] < 2 or b["n_seeds"] < 2:
            s = float("nan")
        axis = ("B" if eid[:3] in ("T01", "T02") else "kết hợp" if eid.startswith("C")
                else (r["desc"].split("_")[1][0] if "_" in r["desc"] else ""))
        rows.append({
            "exp_id": eid, "backbone": r["backbone"], "axis": AXES.get(axis, axis),
            "diff_vs_baseline": diff_vs(first_cfg[eid], first_cfg[base]), "baseline": base,
            "seeds": r["seeds"], "n_seeds": r["n_seeds"],
            "macro_f1_val_mean": r["macro_f1_val_mean"], "macro_f1_val_std": r["macro_f1_val_std"],
            "top1_val_mean": r["top1_val_mean"], "baseline_macro_f1_val": b["macro_f1_val_mean"],
            "delta_macro_f1_vs_baseline": delta, "max_std": s,
            "verdict": ("< 2 seed ở một bên: chưa kết luận" if np.isnan(s) else
                        "tốt hơn (Δ > std)" if delta > s else "kém hơn (Δ < -std)" if delta < -s
                        else "không phân biệt được (|Δ| ≤ std)"),
            "f1_chinee_val": r["f1_chinee_val_mean"], "f1_snake_val": r["f1_snake_val_mean"],
            "best_epoch_mean": r["best_epoch_mean"], "curve": f"curves/{eid}_{r['desc']}.png",
        })
    return pd.DataFrame(rows)


def backbones_sheet(runs: pd.DataFrame) -> pd.DataFrame:
    b = agg(runs[runs.exp_id.str.fullmatch(r"B\d\d")])
    sweep = RUNS / "res_sweep.csv"
    if sweep.exists() and len(b):
        sw = pd.read_csv(sweep)
        best = sw.loc[sw.groupby(["exp_id", "seed"])["macro_f1_val"].idxmax()]
        bm = best.groupby("exp_id").agg(best_test_res=("test_res", lambda x: "/".join(map(str, sorted(set(x))))),
                                        macro_f1_val_best_res=("macro_f1_val", "mean"))
        b = b.merge(bm, left_on="exp_id", right_index=True, how="left")
    b["curve"] = "curves/" + b["exp_id"] + "_" + b["desc"] + ".png"
    return b


def inference_sheets():
    inf, lat = [], []
    for d in sorted((RUNS / "inference").glob("*")):
        if (d / "inference_val.csv").exists():
            inf.append(pd.read_csv(d / "inference_val.csv").assign(tag=d.name))
        if (d / "latency.csv").exists():
            lat.append(pd.read_csv(d / "latency.csv").assign(tag=d.name))
    return (pd.concat(inf) if inf else pd.DataFrame({"note": ["chưa có (run_inference.py)"]}),
            pd.concat(lat) if lat else pd.DataFrame({"note": ["chưa có"]}))


def final_sheets(ids: list[str]):
    """Final + PerClass tính bằng eval.load_group từ predictions/ (cùng định nghĩa với lúc chấm)."""
    pred = SUB / "predictions"
    names = ev.load_names(str(LABELS / "labels.csv"))
    final_rows, pc_rows = [], []
    for eid in ids:
        tests = sorted(pred.glob(f"{eid}_seed*_test.csv"))
        if not tests:
            continue
        g = ev.load_group([str(p) for p in tests], str(LABELS / "test_subset0.csv"))
        lock = {}
        for p, m in zip(g.preds, g.metrics):
            vf = pred / f"{eid}_seed{p.seed}_val.csv"
            mv = ev.compute_metrics(*(lambda q: (q.y_true, q.y_pred, q.probs))(ev.read_pred(str(vf)))) \
                if vf.exists() else {}
            lk = RUNS / "final" / f"{eid}_seed{p.seed}.json"
            lock = json.loads(lk.read_text()) if lk.exists() else {}
            final_rows.append({"exp_id": eid, "seed": p.seed,
                               "config": f"{lock.get('backbone')} | run {lock.get('run')} | "
                                         f"suy luận {lock.get('method')}{' + TS' if lock.get('temperature_scaling') else ''}",
                               "macro_f1_val": mv.get("macro_f1"), "macro_f1_test": m["macro_f1"],
                               "top1_test": m["top1"], "balanced_acc_test": m["balanced_acc"], "ece_test": m["ece"],
                               **{f"recall_{c.split()[0].lower()}_test": m["recall"][i] for c, i in HARD.items()}})
        s = g.summary
        final_rows.append({"exp_id": eid, "seed": f"mean ± std ({len(g.preds)} seed)",
                           "macro_f1_test": ev.fmt(*s["macro_f1"]), "top1_test": ev.fmt(*s["top1"]),
                           "balanced_acc_test": ev.fmt(*s["balanced_acc"]), "ece_test": ev.fmt(*s["ece"]),
                           **{f"recall_{c.split()[0].lower()}_test": ev.fmt(s["recall"][0][i], s["recall"][1][i])
                              for c, i in HARD.items()}})
        for i, n in enumerate(names):
            pc_rows.append({"config": eid, "class": n, "n_test": int(g.metrics[0]["support"][i]),
                            **{f"{k}_mean": s[k][0][i] for k in ("precision", "recall", "f1")},
                            **{f"{k}_std": s[k][1][i] for k in ("precision", "recall", "f1")}})
    return (pd.DataFrame(final_rows) if final_rows else pd.DataFrame({"note": ["chưa chạy chung kết"]}),
            pd.DataFrame(pc_rows) if pc_rows else pd.DataFrame({"note": ["chưa chạy chung kết"]}))


def extra_sheets() -> dict:
    """Sheet bổ sung: chọn phương pháp suy luận qua 3 seed (val), ONNX, lệch phân phối (điểm thưởng)."""
    out = {}
    ms = [pd.read_csv(f).assign(config=f.stem.replace("MS_", "").replace("_summary", ""))
          for f in sorted((RUNS / "inference").glob("MS_*_summary.csv"))]
    if ms:
        out["MethodSelect"] = pd.concat(ms, ignore_index=True)
    onnx = [pd.read_csv(f) for f in sorted((RUNS / "onnx").glob("onnx_*.csv")) if not f.stem.startswith("onnx_smoke")]
    if onnx:
        out["ONNX"] = pd.concat(onnx, ignore_index=True)
    rob = [pd.read_csv(f) for f in sorted((RUNS / "robustness").glob("rob_*.csv"))]
    if rob:
        out["Robustness"] = pd.concat(rob, ignore_index=True)
    return out


FINAL_DESC = {
    "T00": ("mốc Swin-T", "Swin-T, công thức nền T00 + suy luận I00"),
    "F01": ("chung kết: tốt nhất (ngoại tuyến)", "Swin-T C01s (res 288, 20 epoch) + 10 crop gộp logit + TS"),
    "T00m": ("mốc MobileNetV3", "MobileNetV3-L, công thức nền T00 + suy luận I00"),
    "F02": ("chung kết: mạng nhẹ", "MobileNetV3-L C03m (LR x3, 20 epoch, TrivialAugment, res 288) + 5 crop + TS"),
    "F03": ("chung kết: thời gian thực", "MobileNetV3-L C03m + KD từ Swin (a 0.9, T 4) + 1 view 352 + TS"),
}


def summary_sheet(runs: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """Bảng một trang: (A) chung kết vs mốc trên TEST; (B) top 10 cấu hình huấn luyện theo macro-F1 VAL
    (chỉ cấu hình có >= 3 seed), kèm GMAC (chi phí không phụ thuộc phần cứng)."""
    rows = []
    agg_f = final[final["seed"].astype(str).str.startswith("mean")].set_index("exp_id") if "seed" in final else None
    for eid, (kind, desc) in FINAL_DESC.items():
        if agg_f is None or eid not in agg_f.index:
            continue
        r = agg_f.loc[eid]
        vals = final[(final.exp_id == eid) & ~final["seed"].astype(str).str.startswith("mean")]
        rows.append({"phần": "A. chung kết vs mốc (test, 3 seed)", "exp_id": eid, "vai trò": kind, "mô tả": desc,
                     "macro_f1_val": f"{vals.macro_f1_val.mean():.4f} ± {vals.macro_f1_val.std(ddof=1):.4f}",
                     "macro_f1_test": r["macro_f1_test"], "top1_test": r["top1_test"], "ece_test": r["ece_test"],
                     "recall_chinee_test": r["recall_chinee_test"], "recall_snake_test": r["recall_snake_test"]})
    a = agg(runs)
    a = a[(a.n_seeds >= 3) & ~a.exp_id.str.fullmatch(r"F\d\d")].sort_values("macro_f1_val_mean", ascending=False).head(10)
    for _, r in a.iterrows():
        rows.append({"phần": "B. top 10 cấu hình huấn luyện (val, >= 3 seed)", "exp_id": r.exp_id,
                     "vai trò": r.backbone, "mô tả": r.desc,
                     "macro_f1_val": f"{r.macro_f1_val_mean:.4f} ± {r.macro_f1_val_std:.4f}",
                     "GMAC": round(r.gmacs, 3), "n_seeds": r.n_seeds})
    return pd.DataFrame(rows)


def write_xlsx(sheets: dict, path: Path):
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df = df.drop(columns=[c for c in ("cfg",) if c in df]).reset_index(drop=True)
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "B2"
            for j, col in enumerate(df.columns, 1):
                ws.cell(1, j).font = Font(bold=True)
                ws.column_dimensions[get_column_letter(j)].width = min(45, max(10, len(str(col)) + 2))
                if pd.api.types.is_float_dtype(df[col]):
                    for i in range(2, len(df) + 2):
                        ws.cell(i, j).number_format = "0.0000"
            if name == "Summary" and "exp_id" in df:
                for i in df.index[df.exp_id.isin(["F01"])]:
                    for j in range(1, len(df.columns) + 1):
                        ws.cell(i + 2, j).fill = PatternFill("solid", fgColor="FFF2CC")
                for j in range(1, len(df.columns) + 1):
                    ws.column_dimensions[get_column_letter(j)].width = 18 if df.columns[j - 1] != "mô tả" else 70
                continue
            key = next((c for c in ("macro_f1_val_mean", "macro_f1_val") if c in df), None)
            if key and df[key].notna().any() and pd.api.types.is_numeric_dtype(df[key]):
                best = int(df[key].astype(float).idxmax())        # chỉ số dòng 0..n-1 sau reset_index
                for j in range(1, len(df.columns) + 1):
                    ws.cell(best + 2, j).fill = PatternFill("solid", fgColor="FFF2CC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", nargs="*", default=[], help="exp_id chung kết, vd F01 F02")
    ap.add_argument("--baseline", nargs="*", default=["T00"], help="exp_id mốc, vd T00 T00m")
    ap.add_argument("--out", default=str(SUB / "results.xlsx"))
    args = ap.parse_args()
    runs = load_runs()
    backbones = backbones_sheet(runs)
    training = training_sheet(runs)
    inference, latency = inference_sheets()
    final, perclass = final_sheets(args.final + args.baseline)
    sheets = {"Summary": summary_sheet(runs, final), "Backbones": backbones,
              "Training": training, "Inference": inference, "Final": final, "PerClass": perclass,
              "Latency": latency, **extra_sheets()}
    write_xlsx(sheets, Path(args.out))
    for k, v in sheets.items():
        print(f"{k:10s} {len(v)} dòng")
    print("Đã ghi", args.out)


if __name__ == "__main__":
    main()
