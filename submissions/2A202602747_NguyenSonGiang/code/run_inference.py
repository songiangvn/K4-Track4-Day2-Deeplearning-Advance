"""run_inference.py - Bước 3: so sánh phương pháp suy luận trên VAL (không huấn luyện lại).

    python run_inference.py --run T00/seed0 [--ensemble B03/seed0 B01/seed0] [--ema-run T12/seed0] \
        [--tag I_T00] [--no-latency]

Đầu vào: một lần chạy đã train (runs/<exp_id>/seed<k>/: config.json, best.pt, val_logits.npy).
Đầu ra (runs/inference/<tag>/): inference_val.csv (một dòng mỗi phương pháp), latency.csv,
probs_val.npz (xác suất val của mọi phương pháp); ảnh figures/<tag>_tradeoff.png.

Phương pháp (GUIDE mục 4); S = img_size lúc train của lần chạy (224 hoặc 288):
  I00   1 view: đúng tiền xử lý val lúc train (mốc), ở kích thước S
  I01   TTA lật ngang, K=2                        I01L  như I01 nhưng gộp logit (I03)
  I02a  5 crop S từ ảnh phóng lên R = 9S/7 (288 nếu S=224), K=5     I02aL gộp logit
  I02b  10 crop (5 crop + lật), K=10              I02bL gộp logit
  I02c  3 tỉ lệ (S, S+64, S+96) của cả ảnh, K=3   I02cL gộp logit
  Mọi view đều QUA NỘI SUY (không lấy crop/kích thước 256 trực tiếp từ ảnh gốc 256): ảnh không nội suy
  lệch độ sắc nét với ảnh train và làm sụp mạng BN (xem BASE2 trong run_exps.py; bản đầu của I02a
  cắt 224 thẳng từ ảnh 256 cho MobileNetV3 macro-F1 val 0.786 so với 0.925 của I00).
  I04_c<r>  độ phân giải kiểm tra r, center-crop tỉ lệ 0.875 (Resize r/0.875 rồi CenterCrop r)
  I04_f<r>  độ phân giải kiểm tra r, ảnh gốc resize về r (không crop)
  I05   ensemble trung bình xác suất của --run và các --ensemble (mỗi model 1 view)
  I06   trọng số EMA (lần chạy --ema-run train có EMA; 1 view, không tốn thêm khi suy luận)
  I07   I00 + temperature scaling. ECE val đo NGOÀI MẪU bằng khớp chéo 2 phần trên val;
        T cuối cùng (để áp sang test) khớp trên toàn bộ val
  I08a  gộp BN vào conv, FP32     I08b  FP16 (model.half())     I08c  AMP (autocast)
  I08d  gộp BN + FP16
Độ trễ: chỉ forward mạng (không tính đọc/giải mã ảnh), đầu vào đã ở trên GPU; TTA/multi-crop
gộp K view thành MỘT batch K ảnh; warmup 20, 100 lần đo; batch 1 và batch 32. TF32 TẮT: "fp32" là FP32 thật.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import dataset as D
import inference as I
from benchmark import latency_report

HERE = Path(__file__).resolve().parent
SUB = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))       # eval.py của repo gốc
RUNS = REPO / "runs"
IMAGES, LABELS = REPO / "data" / "images", REPO / "data" / "labels"
HARD = {"Chinee Apple": 0, "Snake Weed": 7}


class NotApplicable(Exception):
    """Phương pháp suy luận không áp dụng được cho kiến trúc này (ví dụ multi-scale với Swin)."""


# --------------------------------------------------------------------------- #
# Ngữ cảnh: model, loader theo (split, transform), cache logit
# --------------------------------------------------------------------------- #
@dataclass
class Ctx:
    run: Path
    device: torch.device
    num_workers: int = 10
    store: dict = field(default=None)
    models: dict = field(default_factory=dict)
    loaders: dict = field(default_factory=dict)
    cfg: dict = field(default=None)

    def __post_init__(self):
        self.store = D.load_byte_store(IMAGES)
        self.cfg = json.loads((self.run / "config.json").read_text())
        self.splits = dict(zip(("train", "val", "test"), D.load_split(LABELS, self.cfg.get("fold", 0))))

    @property
    def arch(self) -> str:
        b = self.cfg["backbone"]
        return "swin" if "swin" in b else "vit" if ("vit" in b or "deit" in b) else "cnn"

    def model(self, size: int | None = None, run: Path | None = None):
        """Model của `run` (mặc định self.run) chạy được ở kích thước `size`."""
        run = run or self.run
        size = self.img_size if size is None else size
        arch = self.arch if run == self.run else "cnn"
        key = (str(run), size if arch == "swin" else (0 if arch == "cnn" else "any"))
        if key not in self.models:
            self.models[key] = I.load_model(run, self.device, img_size=size if arch == "swin" else None,
                                            any_size=arch == "vit")[0]
        return self.models[key]

    @property
    def img_size(self) -> int:
        return self.cfg.get("img_size", 224)

    @property
    def crop_pct(self) -> float:
        """Tiền xử lý val/test của lần chạy (0.875: Resize 256 + CenterCrop 224; 1.0: resize cả ảnh)."""
        return self.cfg.get("crop_pct", D.CROP_PCT)

    def loader(self, split: str, size: int | None = None, crop_pct: float | None = None):
        """Loader không xáo trộn; mặc định đúng tiền xử lý val của lần chạy (img_size, crop_pct)."""
        size = self.img_size if size is None else size
        crop_pct = self.crop_pct if crop_pct is None else crop_pct
        key = (split, size, crop_pct)
        if key not in self.loaders:
            tf = D.build_transforms(False, size, crop_pct=crop_pct)
            self.loaders[key] = D.make_loader(self.splits[split], IMAGES, tf, 64, False,
                                              num_workers=self.num_workers, cache=self.store,
                                              persistent=False)
        return self.loaders[key]


def multicrop_size(S: int) -> int:
    """Cạnh ảnh phóng to trước khi cắt 5 crop S (khác 256 để luôn có nội suy): 288 khi S = 224."""
    return round(S * 9 / 7)


def multiscale_sizes(S: int) -> tuple[int, ...]:
    return (S, S + 64, S + 96)                    # S=224: 224/288/320 (tránh đúng 256 = ảnh gốc)


def views_for(method: str, crop_pct: float = D.CROP_PCT, S: int = 224):
    """(kích thước model, transform (size, crop_pct), hàm sinh view, K) của mỗi họ phương pháp.

    I00/I01 dùng đúng tiền xử lý val của lần chạy (`crop_pct`, kích thước S). I02 nhận ảnh gốc 256
    (không biến đổi) rồi tự resize trên GPU, nên view nào cũng qua nội suy."""
    full = (256, 1.0)                                        # ảnh gốc 256x256, chưa nội suy
    R = multicrop_size(S)
    return {
        "I00": (S, (S, crop_pct), lambda x: [x], 1),
        "I01": (S, (S, crop_pct), I.views_hflip, 2),
        "I02a": (S, full, lambda x: I.views_multicrop(I.views_multiscale(x, (R,))[0], S), 5),
        "I02b": (S, full, lambda x: I.views_multicrop(I.views_multiscale(x, (R,))[0], S, flip=True), 10),
        "I02c": (None, full, lambda x: I.views_multiscale(x, multiscale_sizes(S)), 3),
    }[method]


def view_logits(ctx: Ctx, family: str, split: str):
    """List K mảng logit [N, 9] của một họ view, kèm (names, y)."""
    size, (tsize, cpct), vf, k = views_for(family, ctx.crop_pct, ctx.img_size)
    if family == "I02c" and ctx.arch == "swin":
        raise NotApplicable("Swin chỉ chạy ở một kích thước mỗi model: bỏ qua multi-scale")
    model = ctx.model(size or max(multiscale_sizes(ctx.img_size)))
    names, y, logits = I.predict_views(model, ctx.loader(split, tsize, cpct), ctx.device, vf)
    assert len(logits) == k
    return names, y, logits


def resolution_logits(ctx: Ctx, split: str, r: int, crop_pct: float):
    model = ctx.model(r)
    return I.predict_logits(model, ctx.loader(split, r, crop_pct), ctx.device)


# --------------------------------------------------------------------------- #
# Chỉ số
# --------------------------------------------------------------------------- #
def metrics(y, probs) -> dict:
    from eval import compute_metrics
    m = compute_metrics(np.asarray(y), probs.argmax(1), probs)
    out = {"macro_f1_val": m["macro_f1"], "top1_val": m["top1"], "balanced_acc_val": m["balanced_acc"],
           "ece_val": m["ece"], "nll_val": m["nll"]}
    for c, i in HARD.items():
        out[f"f1_{c.replace(' ', '_')}"] = float(m["f1"][i])
        out[f"recall_{c.replace(' ', '_')}"] = float(m["recall"][i])
    return out


# --------------------------------------------------------------------------- #
# Độ trễ
# --------------------------------------------------------------------------- #
def tta_builder(vf):
    def builder(m, x):
        def fn():
            v = vf(x)
            if all(t.shape == v[0].shape for t in v):           # cùng kích thước: gộp thành 1 batch
                out = m(torch.cat(v)).float().softmax(-1)
                return out.view(len(v), x.size(0), -1).mean(0)
            return torch.stack([m(t).float().softmax(-1) for t in v]).mean(0)
        return fn
    return builder


def ensemble_builder(models):
    def builder(_, x):
        def fn():
            return torch.stack([mm(x).float().softmax(-1) for mm in models]).mean(0)
        return fn
    return builder


def measure(model, img_size, note, batches=(1, 32), dtype="fp32", builder=None, iters=100):
    rows = []
    for b in batches:
        r = latency_report(model, b, img_size, dtype, warmup=20, iters=iters, fn_builder=builder, note=note)
        rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="vd T00/seed0 (dưới runs/)")
    ap.add_argument("--ensemble", nargs="*", default=[], help="các lần chạy khác để ensemble với --run")
    ap.add_argument("--ema-run", default=None, help="lần chạy cùng công thức nhưng có EMA")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resolutions", nargs="*", type=int, default=None,
                    help="mặc định: 224, 256 và S, S+32, S+64, S+96")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--num-workers", type=int, default=10)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = RUNS / args.run
    tag = args.tag or f"I_{args.run.replace('/', '_')}"
    out = RUNS / "inference" / tag
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(run, dev, args.num_workers)
    S = ctx.img_size
    if args.resolutions is None:
        args.resolutions = sorted({224, 256, S, S + 32, S + 64, S + 96})
    torch.backends.cudnn.benchmark = True
    # FP32 thật: mặc định PyTorch cho conv FP32 trên GPU Ampere chạy bằng TF32 (sai số logit ~1e-2), làm
    # model gộp BN và model gốc lệch dự đoán ở vài ảnh sát ranh giới. Tắt để I08 so FP32/FP16/AMP đúng nghĩa.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model_name = f"{ctx.cfg['exp_id']}/seed{ctx.cfg['seed']} ({ctx.cfg['backbone']})"
    print("Model:", model_name, "| arch:", ctx.arch, flush=True)

    rows, probs_all, lat_rows = [], {}, []
    names_ref = None

    def add(eid, method, k, probs, y, names, note="", model=model_name):
        nonlocal names_ref
        if names_ref is None:
            names_ref = names
        assert list(names) == list(names_ref), f"{eid}: thứ tự ảnh khác I00"
        probs_all[eid] = probs.astype(np.float32)
        rows.append({"exp_id": eid, "method": method, "model": model, "K": k, **metrics(y, probs),
                     "note": note})
        r = rows[-1]
        print(f"{eid:9s} K={k:<2} F1={r['macro_f1_val']:.4f} top1={r['top1_val']:.4f} "
              f"ECE={r['ece_val']:.4f}  {method}", flush=True)

    # I00 / I01 / I02 (+ biến thể gộp logit I03)
    i00_logits = None
    S, R = ctx.img_size, multicrop_size(ctx.img_size)
    i00_desc = (f"1 view (resize cả ảnh 256 -> {S})" if ctx.crop_pct == 1.0
                else f"1 view (Resize {round(S / ctx.crop_pct)} + CenterCrop {S})")
    for fam, desc in [("I00", i00_desc), ("I01", "TTA lật ngang"),
                      ("I02a", f"5 crop {S} từ ảnh phóng {R}"), ("I02b", f"10 crop {S} (5 crop + lật) từ ảnh phóng {R}"),
                      ("I02c", "3 tỉ lệ " + "/".join(map(str, multiscale_sizes(S))) + " (cả ảnh)")]:
        try:
            names, y, lg = view_logits(ctx, fam, "val")
        except NotApplicable as e:
            rows.append({"exp_id": fam, "method": desc, "model": model_name, "note": f"không áp dụng: {e}"})
            print(fam, "bỏ qua:", e)
            continue
        if fam == "I00":
            i00_logits, y_val = lg[0], y
            add("I00", desc, 1, I.softmax(lg[0]), y, names)
            continue
        add(fam, f"{desc}, gộp xác suất", len(lg), I.aggregate_views(lg, "prob"), y, names)
        add(f"{fam}L", f"{desc}, gộp logit (I03)", len(lg), I.aggregate_views(lg, "logit"), y, names)

    # I04: độ phân giải kiểm tra
    for crop_pct, tagc in ((D.CROP_PCT, "c"), (1.0, "f")):
        for r in args.resolutions:
            names, y, lg = resolution_logits(ctx, "val", r, crop_pct)
            how = f"Resize {round(r / crop_pct)} + CenterCrop {r}" if tagc == "c" else f"ảnh gốc resize {r}"
            add(f"I04_{tagc}{r}", f"độ phân giải kiểm tra {r} ({how})", 1, I.softmax(lg), y, names)

    # I05: ensemble (dùng val_logits.npy đã lưu của từng lần chạy: cùng tiền xử lý với I00)
    if args.ensemble:
        members = [run] + [RUNS / e for e in args.ensemble]
        member_probs = []
        for mrun in members:
            mcfg = json.loads((mrun / "config.json").read_text())
            pv = pd.read_csv(Path(mcfg["pred_dir"]) / f"{mcfg['exp_id']}_seed{mcfg['seed']}_val.csv")
            assert pv["Filename"].tolist() == list(names_ref), f"{mrun}: thứ tự val khác"
            assert (mcfg["img_size"], mcfg.get("crop_pct", D.CROP_PCT)) == (S, ctx.crop_pct), \
                f"{mrun}: tiền xử lý khác I00"
            member_probs.append(I.softmax(np.load(mrun / "val_logits.npy")))
        desc = " + ".join(str(m.relative_to(RUNS)) for m in members)
        add("I05", f"ensemble {len(members)} model (TB xác suất)", len(members),
            I.ensemble_probs(member_probs), y_val, names_ref, note=desc, model=desc)

    # I06: EMA
    if args.ema_run:
        erun = RUNS / args.ema_run
        ecfg = json.loads((erun / "config.json").read_text())
        add("I06", f"trọng số EMA (decay {ecfg['ema_decay']})", 1,
            I.softmax(np.load(erun / "val_logits.npy")), y_val, names_ref,
            note="model khác: train có EMA, cùng công thức", model=f"{ecfg['exp_id']}/seed{ecfg['seed']}")

    # I07: temperature scaling
    T_full = I.fit_temperature(i00_logits, y_val)
    p_cf, temps = I.crossfit_temperature(i00_logits, y_val)
    add("I07", "I00 + temperature scaling (ECE ngoài mẫu, khớp chéo 2 phần val)", 1, p_cf, y_val,
        names_ref, note=f"T toàn val = {T_full:.4f}; T từng phần = {[round(t, 4) for t in temps]}")
    rows[-1]["T"] = T_full

    # I08: gộp BN, FP16, AMP
    base = ctx.model(S)
    loader = ctx.loader("val")
    x_ex = next(iter(loader))[0][:8].to(dev)
    fused = I.fuse_conv_bn(base, example=x_ex)
    n_pairs, fuse_err = fused.fused_pairs, getattr(fused, "fuse_max_abs_err", None)
    names, y, lg = I.predict_logits(fused, loader, dev, amp=False)
    add("I08a", "gộp BN vào conv, FP32", 1, I.softmax(lg), y, names,
        note=f"{n_pairs} cặp conv-BN; lệch logit lớn nhất {fuse_err:.2e}" if n_pairs else "không có BN")
    names, y, lg32 = I.predict_logits(base, loader, dev, amp=False)
    add("I08_fp32", "FP32 thuần (không autocast)", 1, I.softmax(lg32), y, names)
    half = copy.deepcopy(base).half()
    names, y, lgh = I.predict_views(half, loader, dev, lambda x: [x.half()], amp=False)
    add("I08b", "FP16 (model.half())", 1, I.softmax(lgh[0]), y, names,
        note=f"lệch logit lớn nhất so với FP32: {np.abs(lgh[0] - lg32).max():.2e}")
    names, y, lga = I.predict_logits(base, loader, dev, amp=True)
    add("I08c", "AMP (autocast FP16)", 1, I.softmax(lga), y, names)
    fused_half = copy.deepcopy(fused).half()
    names, y, lgfh = I.predict_views(fused_half, loader, dev, lambda x: [x.half()], amp=False)
    add("I08d", "gộp BN + FP16", 1, I.softmax(lgfh[0]), y, names)

    df = pd.DataFrame(rows)
    np.savez_compressed(out / "probs_val.npz", filenames=np.array(names_ref), y=y_val, **probs_all)
    (out / "temperature.json").write_text(json.dumps({"T_full_val": T_full, "T_crossfit": temps}))

    # ---- độ trễ ----
    if not args.no_latency and dev.type == "cuda":
        def lat(eid, model, size, dtype="fp32", builder=None, fusedflag=False):
            for r in measure(model, size, eid, dtype=dtype, builder=builder):
                lat_rows.append({"exp_id": eid, "fused_bn": fusedflag, **r})
                print(f"  latency {eid:9s} {dtype} b{r['batch']:<2} p50={r['p50_ms']:.2f} "
                      f"p95={r['p95_ms']:.2f} p99={r['p99_ms']:.2f} ms  {r['images_per_s']:.0f} img/s",
                      flush=True)

        mS = ctx.model(S).float()
        lat("I00", mS, S)
        lat("I01", mS, S, builder=tta_builder(I.views_hflip))
        lat("I02a", mS, 256, builder=tta_builder(views_for("I02a", ctx.crop_pct, S)[2]))
        lat("I02b", mS, 256, builder=tta_builder(views_for("I02b", ctx.crop_pct, S)[2]))
        if ctx.arch != "swin":
            lat("I02c", ctx.model(max(multiscale_sizes(S))), 256,
                builder=tta_builder(views_for("I02c", ctx.crop_pct, S)[2]))
        for r in args.resolutions:
            lat(f"I04_c{r}", ctx.model(r), r)
        if args.ensemble:
            ens = [ctx.model(S)] + [ctx.model(S, RUNS / e) for e in args.ensemble]
            lat("I05", ens[0], S, builder=ensemble_builder(ens))
        lat("I08a", fused, S, fusedflag=True)
        lat("I08b", copy.deepcopy(mS), S, dtype="fp16")
        lat("I08c", mS, S, dtype="amp")
        lat("I08d", copy.deepcopy(fused), S, dtype="fp16", fusedflag=True)
        lat_df = pd.DataFrame(lat_rows)
        lat_df.to_csv(out / "latency.csv", index=False)

        b1 = lat_df[lat_df.batch == 1].set_index("exp_id")
        b32 = lat_df[lat_df.batch == 32].set_index("exp_id")
        same = {"I01L": "I01", "I02aL": "I02a", "I02bL": "I02b", "I02cL": "I02c",
                "I06": "I00", "I07": "I00", "I08_fp32": "I00"}
        # I04_f<r> có cùng chi phí mạng với I04_c<r> (chỉ khác tiền xử lý)
        same.update({f"I04_f{r}": f"I04_c{r}" for r in args.resolutions})
        for col, src in (("p50_ms_b1", "p50_ms"), ("p95_ms_b1", "p95_ms"), ("p99_ms_b1", "p99_ms")):
            df[col] = [b1[src].get(same.get(e, e), np.nan) for e in df.exp_id]
        df["images_per_s_b32"] = [b32["images_per_s"].get(same.get(e, e), np.nan) for e in df.exp_id]
        df["rel_cost_vs_I00"] = df["p50_ms_b1"] / b1.loc["I00", "p50_ms"]
        df["gpu"] = lat_df["gpu"].iloc[0]

    df.to_csv(out / "inference_val.csv", index=False)
    print("Đã ghi", out / "inference_val.csv")
    if "p50_ms_b1" in df:
        plot_tradeoff(df, SUB / "figures" / f"{tag}_tradeoff.png", model_name)


def plot_tradeoff(df, path, title):
    """Vẽ bằng make_figures.fig_tradeoff (một điểm mỗi phương pháp, nhãn không chồng nhau)."""
    import make_figures
    make_figures.fig_tradeoff(Path(path).name.replace("_tradeoff.png", ""))


if __name__ == "__main__":
    main()
