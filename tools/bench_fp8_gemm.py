#!/usr/bin/env python3
"""FP8 W8A8 GEMM throughput on the job's shapes: sgl_kernel.fp8_scaled_mm (what SGLang runs) vs torch._scaled_mm rowwise.

  /opt/sglang/.venv-sglang/bin/python tools/bench_fp8_gemm.py
"""
import torch


def bench(fn, n=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


def main():
    from sgl_kernel import fp8_scaled_mm

    shapes = []
    for m in (264, 1056, 2112, 3168, 8192):                      # LM: decode bs 8 / 32 / 64 / 96 (x33), prefill chunk
        for n, k, name in ((4096, 1536, "lm.qkv"), (1536, 2048, "lm.o"), (9216, 1536, "lm.gate_up"), (1536, 4608, "lm.down")):
            shapes.append((m, n, k, name))
    for m in (4096, 16384):                                      # ViT packed batches
        for n, k, name in ((3072, 1024, "vit.qkv"), (1024, 1024, "vit.proj"), (8192, 1024, "vit.gate_up"), (1024, 4096, "vit.down")):
            shapes.append((m, n, k, name))
    print(f"{'shape':28s} {'sgl ms':>8s} {'TF/s':>6s} {'torch ms':>9s} {'TF/s':>6s} {'max|diff|':>9s}")
    for m, n, k, name in shapes:
        a = (torch.randn(m, k, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        b = (torch.randn(n, k, device="cuda") * 0.5).to(torch.float8_e4m3fn).t()   # [k, n] column-major like SGLang
        sa = torch.rand(m, 1, device="cuda") * 0.01 + 0.001
        sb = torch.rand(n, 1, device="cuda") * 0.01 + 0.001
        flops = 2.0 * m * n * k
        t1 = bench(lambda: fp8_scaled_mm(a, b, sa, sb, torch.bfloat16, None))
        try:
            t2 = bench(lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sb.t().contiguous(), out_dtype=torch.bfloat16))
            o1 = fp8_scaled_mm(a, b, sa, sb, torch.bfloat16, None)
            o2 = torch._scaled_mm(a, b, scale_a=sa, scale_b=sb.t().contiguous(), out_dtype=torch.bfloat16)
            d = float((o1.float() - o2.float()).abs().max())
            ts = f"{t2:9.3f} {flops / t2 / 1e9:6.0f} {d:9.3g}"
        except Exception as e:  # rowwise scaling not supported on this build / arch
            ts = f"  n/a ({type(e).__name__}: {str(e)[:40]})"
        print(f"{name:12s} {m:6d}x{n:5d}x{k:5d} {t1:8.3f} {flops / t1 / 1e9:6.0f} {ts}", flush=True)


if __name__ == "__main__":
    main()
