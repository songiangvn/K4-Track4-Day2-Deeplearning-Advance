"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - KHÔNG tính tiền xử lý (giải mã JPEG, resize): đầu vào là tensor đã nằm sẵn trên GPU.
    Chỉ đo forward của mạng (với TTA: K lượt forward + gộp xác suất).
"""
from __future__ import annotations

import time

import numpy as np
import torch

DTYPES = ("fp32", "amp", "fp16")


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.
    """
    if iters < 50:
        raise ValueError("Cần >= 50 lần đo (GUIDE.md mục 4.1)")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = np.empty(iters)
    for i in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times[i] = (time.perf_counter() - t0) * 1000
    p50, p95, p99 = np.percentile(times, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "mean": float(times.mean()),
            "std": float(times.std(ddof=1)), "n": iters, "warmup": warmup}


def _forward_fn(model, x, dtype: str):
    if dtype == "amp":
        def fn():
            with torch.autocast(device_type=x.device.type, dtype=torch.float16):
                return model(x)
        return fn
    return lambda: model(x)


@torch.inference_mode()
def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, fn_builder=None, note: str = "") -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast fp16, trọng số fp32) | "fp16" (model.half(), đầu vào half).
    Với "fp16", model bị đổi sang half tại chỗ: truyền vào một bản sao nếu còn cần bản fp32.
    `fn_builder(model, x) -> fn` cho phép đo một quy trình tuỳ ý (ví dụ TTA K view).
    Trả về dict ghi thẳng vào sheet `Latency` của results.xlsx.
    """
    if dtype not in DTYPES:
        raise ValueError(f"dtype={dtype!r}; chọn một trong {DTYPES}")
    dev = torch.device(device)
    model = model.to(dev).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        model = model.half()
        x = x.half()
    fn = fn_builder(model, x) if fn_builder is not None else _forward_fn(model, x, dtype)
    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    return {
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "dtype": dtype, "batch": batch_size, "img_size": img_size,
        "p50_ms": r["p50"], "p95_ms": r["p95"], "p99_ms": r["p99"], "mean_ms": r["mean"],
        "images_per_s": batch_size / (r["p50"] / 1000), "iters": r["n"], "warmup": r["warmup"],
        "torch": torch.__version__, "preprocessing_included": False, "note": note,
    }


def tta_latency(model, k_views: int, view_fn=None, **kw) -> dict:
    """Độ trễ của TTA K view: đo thật K lượt forward + softmax + trung bình (slide trang 63).

    `view_fn(x, i)` tạo view thứ i từ batch x (mặc định: chẵn = gốc, lẻ = lật ngang).
    Kết quả có thêm `k_times_single_p50_ms` để so với giả định "K lần một lượt".
    """
    dtype = kw.get("dtype", "fp32")
    view_fn = view_fn or (lambda x, i: x if i % 2 == 0 else torch.flip(x, dims=[-1]))

    def builder(m, x):
        def fn():
            with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=dtype == "amp"):
                return torch.stack([m(view_fn(x, i)).float().softmax(-1) for i in range(k_views)]).mean(0)
        return fn

    single = latency_report(model, **kw)
    r = latency_report(model, fn_builder=builder, note=f"TTA K={k_views}", **kw)
    r["k_views"] = k_views
    r["k_times_single_p50_ms"] = k_views * single["p50_ms"]
    return r
