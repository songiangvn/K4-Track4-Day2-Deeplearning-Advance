"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LOSSES = ("ce", "ls", "focal", "ce_weighted")


def build_criterion(kind: str = "ce", smoothing: float = 0.1, gamma: float = 2.0,
                    alpha=None, weight=None):
    """Trả về hàm loss theo `kind`:
      - "ce"          : cross-entropy
      - "ls"          : CE + label smoothing `smoothing`
      - "focal"       : focal loss `gamma` (alpha tuỳ chọn: vector trọng số lớp)
      - "ce_weighted" : CE có trọng số lớp `weight` (lấy từ class_weights trên TRAIN)
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(smoothing)
    if kind == "focal":
        return FocalLoss(gamma, alpha)
    if kind == "ce_weighted":
        if weight is None:
            raise ValueError("ce_weighted cần `weight`")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(weight, dtype=torch.float32))
    raise ValueError(f"loss={kind!r} không hỗ trợ; chọn một trong {LOSSES}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Tự cài đặt (không dùng tham số label_smoothing của nn.CrossEntropyLoss) để thấy rõ công thức:
    loss = (1 - eps) * NLL(y) + eps * mean_k(-log p_k). eps = 0 cho đúng CE (kiểm tra trong test_code.py).
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải trong [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        uniform = -logp.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * uniform).mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    Trung bình trên batch. alpha: None hoặc vector trọng số theo lớp (alpha_t = alpha[y]).
    gamma = 0 và alpha = None cho đúng cross-entropy (kiểm tra trong test_code.py).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1 - p_t).clamp(min=0) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: w_c ∝ 1 / n_c
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c ∝ (1 - beta) / (1 - beta ** n_c)
      (Cui et al. arXiv:1901.05555)
    Cả hai đều chuẩn hoá để tổng trọng số = số lớp (trung bình 1).
    """
    n = np.asarray(counts, dtype=np.float64)
    if (n <= 0).any():
        raise ValueError("Mọi lớp phải có ít nhất 1 ảnh")
    if beta == 0:
        w = 1.0 / n
    elif 0 < beta < 1:
        w = (1 - beta) / (1 - np.power(beta, n))
    else:
        raise ValueError("beta phải trong [0, 1)")
    w = w / w.sum() * len(n)
    return torch.as_tensor(w, dtype=torch.float32)


def rand_bbox(h: int, w: int, lam: float, rng: np.random.Generator):
    """Hộp CutMix có diện tích (1 - lam) * h * w, tâm ngẫu nhiên, bị cắt theo biên ảnh."""
    cut = np.sqrt(1.0 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = rng.integers(h), rng.integers(w)
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    return int(y1), int(y2), int(x1), int(x2)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix", rng: np.random.Generator | None = None):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha), perm là hoán vị ngẫu nhiên của batch
    - mixup : x_mix = lam * x + (1 - lam) * x[perm]
    - cutmix: dán hộp từ x[perm] vào x, rồi đặt lại lam = 1 - (diện tích thực của hộp) / (h * w),
              vì hộp bị cắt ở biên có thể nhỏ hơn dự kiến
    - trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]; lam là float
    `rng` mặc định lấy từ numpy global (đã seed bởi set_seed / seed_worker).
    """
    rng = rng if rng is not None else np.random.default_rng(np.random.randint(2 ** 31))
    lam = float(rng.beta(alpha, alpha))
    perm = torch.as_tensor(rng.permutation(x.size(0)), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        y1, y2, x1, x2 = rand_bbox(h, w, lam, rng)
        x_mix = x.clone()
        x_mix[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    else:
        raise ValueError(f"mode={mode!r} không hỗ trợ; chọn 'mixup' hoặc 'cutmix'")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    Accuracy trên batch đã trộn không còn nghĩa bình thường; đánh giá bằng val.
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
