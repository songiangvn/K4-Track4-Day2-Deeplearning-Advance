"""run_exps.py - danh sách thí nghiệm (exp_id -> Config) và chạy theo nhóm.

    python run_exps.py B                    # Bước 1: mọi backbone, công thức nền, seed 0
    python run_exps.py B --only B01 B03     # chỉ một vài exp_id
    python run_exps.py B --skip_existing    # bỏ qua lần chạy đã có summary.json
    python run_exps.py --list               # in danh sách

Mọi thí nghiệm đi qua train.run(Config(...)); file này chỉ khai báo khác biệt so với công thức nền.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train import Config, run, run_dir  # noqa: F401  (run dùng trong process con)

HERE = Path(__file__).resolve().parent
SUB = HERE.parent                       # submissions/<mssv>_<ten>/
REPO = HERE.parents[2]
PATHS = dict(
    images_dir=str(REPO / "data" / "images"),
    labels_dir=str(REPO / "data" / "labels"),
    out_dir=str(REPO / "runs"),         # checkpoint, logit, history (không commit)
    pred_dir=str(REPO / "runs" / "predictions_val"),
    curves_dir=str(SUB / "curves"),
)
COMMON = dict(num_workers=10)

# ---- Bước 1: backbone (GUIDE mục 2). Cùng công thức nền T00, cùng seed 0. ----
# Tag trọng số: đều chỉ tiền huấn luyện trên ImageNet-1k (convnext_tiny mặc định của timm là
# in12k_ft_in1k nên chỉ định rõ fb_in1k để so sánh công bằng hơn).
BACKBONES = {
    "B01": ("resnet50", "resnet50"),
    "B02": ("resnext50_32x4d", "resnext50"),
    "B03": ("convnext_tiny.fb_in1k", "convnext_tiny"),
    "B04": ("deit_small_patch16_224", "deit_small"),
    "B05": ("swin_tiny_patch4_window7_224", "swin_tiny"),
    "B06": ("efficientnet_b0", "efficientnet_b0"),
    "B07": ("mobilenetv3_large_100", "mobilenetv3_large"),
}

# ---- Ablation trục B (augmentation): scale nhỏ nhất của RandomResizedCrop (công thức nền GỐC) ----
# Giả thuyết ban đầu sau Bước 1: mạng BN tăng mạnh macro-F1 val khi tăng độ phân giải test (MobileNetV3
# 0.78 -> 0.95 ở 320), nghi lệch tỉ lệ train/test do RandomResizedCrop scale=(0.08, 1).
# Kết quả (3 seed): chỉ giúp MobileNetV3 (+0.056 ở 0.5), không giúp ResNet-50/ConvNeXt -> không phải
# nguyên nhân chính. Chẩn đoán tiếp (val): nguyên nhân là ĐỘ SẮC NÉT - ảnh train luôn bị nội suy bởi
# RandomResizedCrop, còn val "Resize 256 + CenterCrop 224" trên ảnh gốc 256 thì không nội suy; chỉ cần
# GaussianBlur sigma 0.5 lúc val là MobileNetV3 lên 0.93. Xem BASE2.
# Mốc của từng backbone là chính lần chạy B tương ứng (cùng công thức nền, scale_min 0.08).
# 3 backbone đại diện: ConvNeXt-T (LayerNorm), MobileNetV3-L (nhẹ, BN), ResNet-50 (mốc, BN).
SCALE_BACKBONES = {"r": "B01", "c": "B03", "m": "B07"}
SCALES = {"T01": 0.25, "T02": 0.5}
SCALE_GROUP = {}
for suffix, bid in SCALE_BACKBONES.items():
    bb, name = BACKBONES[bid]
    for tid, smin in SCALES.items():
        SCALE_GROUP[f"{tid}{suffix}"] = dict(backbone=bb, rrc_scale_min=smin, desc=f"{name}_rrc{smin}")

GROUPS: dict[str, dict[str, dict]] = {
    "B": {eid: dict(backbone=bb, desc=desc) for eid, (bb, desc) in BACKBONES.items()},
    "S": SCALE_GROUP,
}

# ---- Điểm thưởng: DINOv2 ViT-S/14 (RUBRIC mục 2) ----
# Cùng công thức nền BASE2 với B11-B17 để so trực tiếp với CNN tinh chỉnh. Linear probe: backbone đóng
# băng, chỉ train head; thử 2 LR head (chọn trên val). X03: tinh chỉnh toàn bộ, cùng LR như mọi backbone.
DINO = "vit_small_patch14_dinov2.lvd142m"
X_GROUP = {
    "X01": dict(backbone=DINO, init="frozen", lr_head=1e-3, desc="dinov2_vits14_linear_probe_lr1e-3"),
    "X02": dict(backbone=DINO, init="frozen", lr_head=1e-2, desc="dinov2_vits14_linear_probe_lr1e-2"),
    "X03": dict(backbone=DINO, desc="dinov2_vits14_finetune"),
}

# ---- Điểm thưởng: chưng cất tri thức (teacher lớn -> student MobileNetV3), khai báo sau khi chốt ----
# Chốt trên val: teacher = C01s/seed0 (Swin-T res 288 + 20 epoch, cấu hình Swin tốt nhất; C02s không
# phân biệt được nên chọn bản đơn giản hơn). Student = MobileNetV3-L với ĐÚNG công thức C03m (cấu hình
# thời gian thực tốt nhất), nên mốc so sánh của X10-X12 là C03m: KD có cải thiện model triển khai không?
KD_TEACHER: str | None = "C01s/seed0"
KD_STUDENT_BASE = "C03m"


def kd_group(teacher: str, student_kw: dict, name: str) -> dict[str, dict]:
    """Student chỉ khác mốc KD_STUDENT_BASE ở việc thêm loss KD (alpha, nhiệt độ T)."""
    kw = {k: v for k, v in student_kw.items() if k != "desc"}
    return {
        "X10": dict(**kw, teacher_run=teacher, kd_alpha=0.5, kd_temperature=4.0, desc=f"{name}_kd_a0.5_T4"),
        "X11": dict(**kw, teacher_run=teacher, kd_alpha=0.9, kd_temperature=4.0, desc=f"{name}_kd_a0.9_T4"),
        "X12": dict(**kw, teacher_run=teacher, kd_alpha=0.5, kd_temperature=1.0, desc=f"{name}_kd_a0.5_T1"),
    }


# ---- Công thức nền mới (cập nhật SAU khi có kết quả nhóm S, chỉ dựa trên val) ----
# None = chưa chốt: nhóm B2 và T bị khoá để không chạy nhầm khi chưa có quyết định.
# Chốt 2026-10-05 trên VAL: tiền xử lý val/test = resize CẢ ảnh 256 -> 224 (crop_pct=1.0) thay vì
# CenterCrop 224 không nội suy, để ảnh đánh giá cũng qua nội suy như ảnh train. Áp cho MỌI backbone.
# Bằng chứng (val, checkpoint B gốc, seed 0): MobileNetV3 0.783 -> 0.920, ResNeXt 0.714 -> 0.895,
# EfficientNet 0.831 -> 0.924, ResNet-50 0.807 -> 0.845, ConvNeXt 0.952 -> 0.961.
BASE2: dict | None = dict(crop_pct=1.0)

# ---- Bước 1 lặp lại với công thức nền mới: mọi backbone, B11..B17 ----
B2_GROUP = {f"B1{eid[-1]}": dict(backbone=bb, desc=f"{desc}_base2") for eid, (bb, desc) in BACKBONES.items()}


def t_group(suffix: str, backbone: str, name: str) -> dict[str, dict]:
    """Bước 2 (GUIDE mục 3) cho một backbone: mỗi thí nghiệm khác T00<suffix> đúng MỘT yếu tố.

    Trục: A khởi tạo · B augmentation · C loss · D sampler · E LR/optimizer · F chính quy hoá ·
    G độ phân giải/thời gian. (T01/T02 = trục B scale của RandomResizedCrop, nhóm S.)
    """
    exps = {
        "T00": ({}, "baseline"),
        # A. khởi tạo
        "T03": (dict(init="scratch"), "A_scratch"),
        "T04": (dict(init="frozen"), "A_frozen_linear_probe"),
        # B. augmentation
        "T05": (dict(aug="color"), "B_colorjitter"),
        "T06": (dict(aug="vflip"), "B_vflip"),
        "T07": (dict(aug="trivial"), "B_trivialaugment"),
        "T08": (dict(aug="randaug"), "B_randaugment"),
        "T09": (dict(mix="mixup", mix_alpha=0.2), "B_mixup0.2"),
        "T10": (dict(mix="cutmix", mix_alpha=1.0), "B_cutmix1.0"),
        # C. loss
        "T11": (dict(loss="ls", label_smoothing=0.1), "C_labelsmooth0.1"),
        "T12": (dict(loss="focal", focal_gamma=2.0), "C_focal2"),
        "T13": (dict(loss="ce_weighted", class_weight_beta=0.0), "C_ce_weight_inv"),
        "T14": (dict(loss="ce_weighted", class_weight_beta=0.999), "C_ce_classbalanced0.999"),
        # D. cân bằng mẫu
        "T15": (dict(sampler="balanced"), "D_balanced_sampler"),
        # E. LR và optimizer
        "T16": (dict(lr_head=1e-4), "E_same_lr"),
        "T17": (dict(lr_backbone=3e-4, lr_head=3e-3), "E_lr_x3"),
        "T18": (dict(lr_backbone=3e-5, lr_head=3e-4), "E_lr_div3"),
        "T19": (dict(optimizer="sgd", lr_backbone=1e-2, lr_head=1e-1, weight_decay=1e-4), "E_sgd"),
        "T20": (dict(warmup_epochs=0.0), "E_no_warmup"),
        # F. chính quy hoá
        "T21": (dict(ema_decay=0.999), "F_ema0.999"),
        "T22": (dict(drop_path_rate=0.1), "F_droppath0.1"),
        "T23": (dict(weight_decay=0.0), "F_no_weight_decay"),
        # G. độ phân giải và thời gian huấn luyện
        # 288 chứ không phải 256: ảnh gốc đã là 256 nên val ở 256 sẽ không qua nội suy (lặp lại đúng lỗi
        # độ sắc nét đã sửa ở BASE2). Ở 288, val = phóng 256 -> 288, có nội suy như ảnh train.
        "T24": (dict(img_size=288), "G_res288"),
        "T25": (dict(epochs=20), "G_20epochs"),
        # B. augmentation: scale của RandomResizedCrop, lặp lại dưới công thức nền mới
        "T26": (dict(rrc_scale_min=0.25), "B_rrc0.25"),
        "T27": (dict(rrc_scale_min=0.5), "B_rrc0.5"),
    }
    return {f"{tid}{suffix}": dict(backbone=backbone, desc=f"{name}_{desc}", **kw)
            for tid, (kw, desc) in exps.items()}


def c_group(suffix: str, backbone: str, name: str) -> dict[str, dict]:
    """Kết hợp các yếu tố thắng rõ (Δ > std, 3 seed) ở Bước 2, thêm dần theo thứ tự hiệu ứng giảm dần
    (tham lam theo trục) để xem hiệu ứng có cộng dồn không. Chọn hoàn toàn trên val.

    Swin-T: chỉ T24 (res 288, +0.0038) và T25 (20 epoch, +0.0037) vượt std; C02 thêm T14
            (class-balanced, +0.0028, chưa vượt std) để kiểm tra một yếu tố "gần ngưỡng".
    MobileNetV3-L: T17 (LR x3, +0.022) > T25 (20 epoch, +0.014) > T07 (TrivialAugment, +0.007)
            > T24 (res 288, +0.005) ≈ T14 (class-balanced, +0.004). (T19 SGD cùng là đổi LR/optimizer
            với T17 nên không ghép.)
    """
    combos = {
        "s": {
            "C01": (dict(img_size=288, epochs=20), "COMBO_res288_ep20"),
            "C02": (dict(img_size=288, epochs=20, loss="ce_weighted", class_weight_beta=0.999),
                    "COMBO_res288_ep20_cb0.999"),
        },
        "m": {
            "C01": (dict(lr_backbone=3e-4, lr_head=3e-3, epochs=20), "COMBO_lrx3_ep20"),
            "C02": (dict(lr_backbone=3e-4, lr_head=3e-3, epochs=20, aug="trivial"), "COMBO_lrx3_ep20_trivial"),
            "C03": (dict(lr_backbone=3e-4, lr_head=3e-3, epochs=20, aug="trivial", img_size=288),
                    "COMBO_lrx3_ep20_trivial_res288"),
            "C04": (dict(lr_backbone=3e-4, lr_head=3e-3, epochs=20, aug="trivial", loss="ce_weighted",
                         class_weight_beta=0.999), "COMBO_lrx3_ep20_trivial_cb0.999"),
        },
    }[suffix]
    return {f"{cid}{suffix}": dict(backbone=backbone, desc=f"{name}_{desc}", **kw) for cid, (kw, desc) in combos.items()}


# Backbone đi tiếp sang Bước 2/3 (chốt sau nhóm S và B2, dựa trên val): (hậu tố, tên timm, tên ngắn)
# Chốt 2026-10-05 trên val (B11-B17, 3 seed): Swin-T có macro-F1 cao nhất (0.9683 ± 0.0032, hơn ConvNeXt-T và
# DeiT-S quá 1 std); MobileNetV3-L là mạng nhẹ (0.22 GMAC), hoà với EfficientNet-B0 (0.9276 vs 0.9289,
# |Δ| < std) nhưng ít hơn 43% GMAC và train nhanh hơn 25%.
T_BACKBONES: list[tuple[str, str, str]] = [
    ("s", "swin_tiny_patch4_window7_224", "swin_tiny"),
    ("m", "mobilenetv3_large_100", "mobilenetv3_large"),
]

if BASE2 is not None:
    GROUPS["B2"] = {k: {**BASE2, **v} for k, v in B2_GROUP.items()}
    GROUPS["X"] = {k: {**BASE2, **v} for k, v in X_GROUP.items()}
    for suffix, bb, name in T_BACKBONES:
        GROUPS[f"T{suffix}"] = {k: {**BASE2, **v} for k, v in t_group(suffix, bb, name).items()}
    for suffix, bb, name in T_BACKBONES:
        GROUPS[f"C{suffix}"] = {k: {**BASE2, **v} for k, v in c_group(suffix, bb, name).items()}
    # ---- Bước 4: chung kết (GUIDE mục 5). Cấu hình chốt trên val, train lại với seed MỚI (3, 4, 5) để
    # macro-F1 val của chung kết không bị lạc quan do chính các seed đã dùng để chọn cấu hình. ----
    # F01: tốt nhất về độ chính xác (ngoại tuyến) = C01s. F02: cho thời gian thực = C03m.
    GROUPS["F"] = {
        "F01": {**GROUPS["Cs"]["C01s"], "desc": "final_swin_tiny_res288_ep20"},
        "F02": {**GROUPS["Cm"]["C03m"], "desc": "final_mobilenetv3_lrx3_ep20_trivial_res288"},
    }
    if KD_TEACHER:
        student = GROUPS["Cm"][KD_STUDENT_BASE]
        GROUPS["KD"] = {k: {**BASE2, **v} for k, v in
                        kd_group(KD_TEACHER, student, f"{BACKBONES['B07'][1]}_C03").items()}
        # F03: MobileNetV3 + KD = đúng cấu hình X11 (alpha 0.9, T 4), chốt trên val (0.9701 ± 0.0026, hơn C03m
        # quá 1 std) TRƯỚC khi chạy test bất kỳ cấu hình nào. Không train lại: chung kết dùng thẳng 3 seed
        # X11/seed0-2 (final_predict --run X11/seed<k> --exp-id F03). Hệ quả: macro-F1 val của F03 hơi lạc
        # quan (X11 được chọn trong X10-X12 trên chính các seed này); test không bị ảnh hưởng. Với F01/F02,
        # train lại bằng seed mới cho val gần như y hệt (0.9759 vs 0.9757; 0.9639 vs 0.9642).
    # Đa fold (điểm thưởng +3) không làm: tổng điểm thưởng tối đa +10 đã đủ từ các mục khác.


def make_config(group: str, exp_id: str, seed: int = 0) -> Config:
    kw = dict(GROUPS[group][exp_id])
    return Config(exp_id=exp_id, seed=kw.pop("seed", seed), **COMMON, **PATHS, **kw)


def _child(cfg: Config) -> None:
    run(cfg)


def run_isolated(cfg: Config) -> int:
    """Chạy train.run(cfg) trong một process con (spawn) và trả về exit code.

    Mỗi thí nghiệm một process: mọi file descriptor (pipe của DataLoader), bộ nhớ GPU và trạng thái
    cuDNN được hệ điều hành thu hồi khi process kết thúc, nên không tích luỹ qua hàng chục lần chạy
    (trước đây: "OSError: Too many open files" sau ~20 lần chạy MobileNetV3 trong một job).
    """
    import multiprocessing as mp
    p = mp.get_context("spawn").Process(target=_child, args=(cfg,), name=f"{cfg.exp_id}_seed{cfg.seed}")
    p.start()
    p.join()
    return p.exitcode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="*")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for g, exps in GROUPS.items():
            for eid, kw in exps.items():
                print(g, eid, kw)
        return

    failed = []
    for g in args.groups:
        for eid in GROUPS[g]:
            if args.only and eid not in args.only:
                continue
            for seed in args.seeds:
                cfg = make_config(g, eid, seed)
                if args.skip_existing and (run_dir(cfg) / "summary.json").exists():
                    print(f"bỏ qua {eid} seed{seed} (đã có)", flush=True)
                    continue
                code = run_isolated(cfg)
                if code != 0:
                    print(f"LỖI {eid} seed{seed}: process con thoát với mã {code} (traceback ở trên)", flush=True)
                    failed.append(f"{eid}_seed{seed}")
    print("XONG." + (f" LỖI: {failed}" if failed else ""), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
