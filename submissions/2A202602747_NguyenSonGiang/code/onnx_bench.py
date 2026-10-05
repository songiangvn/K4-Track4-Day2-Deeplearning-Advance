"""onnx_bench.py - xuất ONNX và so độ trễ với PyTorch (điểm thưởng +1).

    python onnx_bench.py --run F02/seed0 [--fuse-bn]

Cần môi trường của bài (requirements.txt: onnx, onnxruntime-gpu, onnxscript).
  1. xuất model (FP32, batch động) bằng torch.onnx.export (opset 17)
  2. kiểm tra đúng: logit ONNX Runtime và PyTorch trên 32 ảnh val, sai số lớn nhất, và macro-F1 val của ONNX
  3. đo độ trễ batch 1 và 32: PyTorch FP32 / FP16; ONNX Runtime CUDA EP FP32; ONNX Runtime CPU EP FP32
     (warmup 20, 100 lần đo, p50/p95/p99; đầu vào đã nằm sẵn trên thiết bị, không tính tiền xử lý).
Đo ORT trên GPU qua IO binding (đầu vào/đầu ra nằm sẵn trên GPU) để công bằng với PyTorch;
run_with_iobinding trả về khi GPU đã chạy xong nên số đo đã đồng bộ.
"""
from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch

import inference as I
import run_inference as RI
from benchmark import bench, latency_report


def ort_latency(sess, batch: int, img: int, device: str, iters: int = 100) -> dict:
    import onnxruntime as ort
    x = np.random.randn(batch, 3, img, img).astype(np.float32)
    name_in, name_out = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    if device == "cuda":
        binding = sess.io_binding()
        binding.bind_ortvalue_input(name_in, ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0))
        binding.bind_output(name_out, "cuda")

        def fn():
            sess.run_with_iobinding(binding)
    else:
        def fn():
            sess.run([name_out], {name_in: x})
    r = bench(fn, warmup=20, iters=iters)
    return {"p50_ms": r["p50"], "p95_ms": r["p95"], "p99_ms": r["p99"], "images_per_s": batch / (r["p50"] / 1000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fuse-bn", action="store_true", help="gộp BN vào conv trước khi xuất")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()          # dùng CUDA/cuDNN đi kèm torch (pip nvidia-*), không cần cài riêng

    dev = torch.device("cuda")
    # FP32 thật cho PyTorch (tắt TF32) để so logit với ONNX Runtime đúng nghĩa
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ctx = RI.Ctx(RI.RUNS / args.run, dev, 8)
    model = I.load_model(ctx.run, dev)[0].float().eval()
    img = ctx.cfg["img_size"]
    loader = ctx.loader("val")
    if args.fuse_bn:
        model = I.fuse_conv_bn(model, example=next(iter(loader))[0][:8].to(dev))
    tag = args.tag or f"onnx_{args.run.replace('/', '_')}{'_fused' if args.fuse_bn else ''}"
    out = RI.RUNS / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}.onnx"
    torch.onnx.export(model, torch.randn(1, 3, img, img, device=dev), str(path), opset_version=17,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    # Số luồng CPU = số lõi SLURM thực sự cấp cho job (mặc định ORT lấy toàn bộ lõi của node và cố gắn
    # luồng vào lõi, bị cgroup từ chối: "pthread_setaffinity_np failed", và chạy quá nhiều luồng).
    so = ort.SessionOptions()
    so.intra_op_num_threads = len(os.sched_getaffinity(0))
    so.inter_op_num_threads = 1
    sess_gpu = ort.InferenceSession(str(path), so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    sess_cpu = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    print(f"ORT CPU: {so.intra_op_num_threads} luồng (số lõi được cấp)", flush=True)
    assert "CUDAExecutionProvider" in sess_gpu.get_providers(), "ORT không có CUDA EP"

    # đúng: so logit và macro-F1 val
    x = next(iter(loader))[0][:32]
    with torch.no_grad():
        ref = model(x.to(dev)).float().cpu().numpy()
    max_err = float(np.abs(ref - sess_gpu.run(None, {"input": x.numpy()})[0]).max())
    ys, zs = [], []
    for xb, yb, _ in loader:
        zs.append(sess_gpu.run(None, {"input": xb.numpy()})[0])
        ys.append(yb.numpy())
    m = RI.metrics(np.concatenate(ys), I.softmax(np.concatenate(zs)))
    _, y_pt, z_pt = I.predict_logits(model, loader, dev, amp=False)
    m_pt = RI.metrics(y_pt, I.softmax(z_pt))
    print(f"ONNX vs PyTorch: lệch logit lớn nhất {max_err:.2e}; macro-F1 val ONNX {m['macro_f1_val']:.4f} "
          f"vs PyTorch FP32 {m_pt['macro_f1_val']:.4f}", flush=True)

    rows = []
    gpu = torch.cuda.get_device_name(0)
    for b in (1, 32):
        for dtype in ("fp32", "fp16"):
            r = latency_report(copy.deepcopy(model), b, img, dtype, warmup=20, iters=100)
            rows.append({"runtime": f"PyTorch {torch.__version__}", "device": gpu, "dtype": dtype, "batch": b,
                         **{k: r[k] for k in ("p50_ms", "p95_ms", "p99_ms", "images_per_s")}})
        for name, sess, d in (("ONNX Runtime CUDA EP", sess_gpu, "cuda"), ("ONNX Runtime CPU EP", sess_cpu, "cpu")):
            r = ort_latency(sess, b, img, d, iters=100 if d == "cuda" or b == 1 else 50)
            rows.append({"runtime": f"{name} {ort.__version__}", "device": gpu if d == "cuda" else "CPU",
                         "dtype": "fp32", "batch": b, **r})
        for r in rows[-4:]:
            print(f"  {r['runtime']:32s} {r['dtype']} b{b:<2} p50={r['p50_ms']:.2f} p95={r['p95_ms']:.2f} ms "
                  f"{r['images_per_s']:.0f} img/s", flush=True)
    df = pd.DataFrame(rows).assign(run=args.run, backbone=ctx.cfg["backbone"], fused_bn=args.fuse_bn,
                                   onnx_max_abs_err=max_err, onnx_macro_f1_val=m["macro_f1_val"],
                                   torch_macro_f1_val=m_pt["macro_f1_val"])
    df.to_csv(out / f"{tag}.csv", index=False)
    print("Đã ghi", out / f"{tag}.csv")


if __name__ == "__main__":
    main()
