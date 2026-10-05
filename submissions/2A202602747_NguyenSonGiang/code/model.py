"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    set_train_mode(model)                                         -> None  (train() nhưng giữ phần đóng băng ở eval)
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import torch
from torch import nn

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số thực sự tải được ghi vào model.weight_tag
# (lấy từ model.pretrained_cfg) và vào config.json của mỗi lần chạy.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}
INITS = ("scratch", "frozen", "finetune")


def needs_img_size(name: str) -> bool:
    """Model timm cần biết kích thước đầu vào lúc tạo: DINOv2 (pos-embed train ở 518) và Swin
    (lưới cửa sổ, attention mask cố định theo img_size)."""
    return "dinov2" in name or "swin" in name


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", drop_path_rate: float | None = None,
                img_size: int = 224):
    """Tạo model phân loại 9 lớp qua timm (head mới khởi tạo ngẫu nhiên).

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : không tải trọng số, huấn luyện toàn bộ
      - "frozen"   : trọng số tiền huấn luyện, đóng băng backbone, chỉ train head (linear probe)
      - "finetune" : trọng số tiền huấn luyện, train toàn bộ
    `init` quyết định có tải trọng số hay không; `pretrained=False` chỉ để tạo model rỗng (test, benchmark).
    Sau khi tạo: model.weight_tag = "<arch>.<tag>" của trọng số đã tải (None nếu scratch).
    Head luôn được khởi tạo lại bằng init_head (giống nhau cho mọi backbone).
    """
    import timm

    if init not in INITS:
        raise ValueError(f"init={init!r} không hỗ trợ; chọn một trong {INITS}")
    load = pretrained and init != "scratch"
    kw = {"drop_path_rate": drop_path_rate} if drop_path_rate is not None else {}
    if needs_img_size(name):
        kw["img_size"] = img_size   # DINOv2 train ở 518: timm nội suy pos-embed về img_size khi tải
    model = timm.create_model(name, pretrained=load, num_classes=num_classes, drop_rate=drop_rate, **kw)
    cfg = model.pretrained_cfg
    model.weight_tag = f"{cfg.get('architecture', name)}.{cfg.get('tag', '')}" if load else None
    model.frozen = False
    init_head(model)
    if init == "frozen":
        freeze_backbone(model)
    return model


def init_head(model, std: float = 0.01) -> None:
    """Khởi tạo lại head giống nhau cho mọi backbone: weight ~ N(0, std), bias = 0.

    Mặc định của timm khác nhau giữa các họ; với EfficientNet/MobileNet (init kiểu Google,
    uniform ±1/sqrt(9)) logit ban đầu lớn nên loss ban đầu ~4 thay vì ln 9 ≈ 2.197.
    Với std nhỏ, logit ban đầu ≈ 0 nên loss ban đầu ≈ ln 9 (kiểm tra pipeline, GUIDE mục 1.3).
    """
    head = model.get_classifier()
    if isinstance(head, nn.Linear):
        nn.init.normal_(head.weight, std=std)
        nn.init.zeros_(head.bias)
    else:
        raise TypeError(f"Head kiểu {type(head).__name__} chưa được hỗ trợ trong init_head")


def _head_param_ids(model) -> set[int]:
    return {id(p) for p in model.get_classifier().parameters()}


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head (model.get_classifier()).

    BatchNorm của backbone đóng băng phải ở eval, nếu không running_mean/var vẫn bị cập nhật theo
    batch train (dù trọng số không đổi) và model lúc eval khác lúc train. Vì model.train() bật lại
    toàn bộ, train loop phải gọi set_train_mode(model) thay vì model.train().
    """
    head = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head
    model.frozen = True


def set_train_mode(model) -> None:
    """model.train(); nếu backbone đóng băng thì đưa mọi module trừ head về eval (BN, dropout)."""
    model.train()
    if getattr(model, "frozen", False):
        model.eval()                    # đệ quy: mọi module (kể cả head) về eval
        model.get_classifier().train()  # rồi chỉ bật lại head


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm, bias và các tham số 1 chiều khác của backbone (ndim <= 1, kể cả pos_embed/cls_token
      của ViT nếu timm đánh dấu no_weight_decay): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head, weight_decay = weight_decay (bias của head: weight_decay = 0)
    Bỏ qua tham số requires_grad == False. Nhóm rỗng bị loại.
    """
    head = _head_param_ids(model)
    skip = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    groups = {"backbone_decay": [], "backbone_no_decay": [], "head_decay": [], "head_no_decay": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head else "backbone"
        no_decay = p.ndim <= 1 or n in skip
        groups[f"{part}_{'no_decay' if no_decay else 'decay'}"].append(p)
    spec = {
        "backbone_decay": (lr_backbone, weight_decay),
        "backbone_no_decay": (lr_backbone, 0.0),
        "head_decay": (lr_head, weight_decay),
        "head_no_decay": (lr_head, 0.0),
    }
    return [{"params": ps, "lr": spec[k][0], "weight_decay": spec[k][1], "name": k}
            for k, ps in groups.items() if ps]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size.

    Công cụ: torch.utils.flop_counter.FlopCounterMode (có sẵn trong PyTorch). Nó đếm FLOPs của
    conv/matmul/attention theo quy ước 1 MAC = 2 FLOPs, nên GMAC = FLOPs / 2 / 1e9. Phép elementwise
    (BN, activation, cộng residual) không được đếm, giống fvcore, nên số có thể thấp hơn vài %.
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        model(x)
    model.train(was_training)
    return counter.get_total_flops() / 2 / 1e9
