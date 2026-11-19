#!/usr/bin/env python3
"""gravity_ocr.fp8_gemm: startup self-check (bitwise vs sgl_kernel.fp8_scaled_mm) + timing on the dispatched shapes.

  /opt/sglang/.venv-sglang/bin/python tools/test_fp8_gemm.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bench(fn, n=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n * 1000


def main():
    from sgl_kernel import fp8_scaled_mm

    from gravity_ocr import fp8_gemm as G

    err = G.self_check(fp8_scaled_mm)
    print(f"[fp8-gemm] self-check: {'OK (bitwise on every rule class)' if not err else err}", flush=True)
    G._ORIG, G._ACTIVE = fp8_scaled_mm, True
    torch.manual_seed(1)
    tot_old = tot_new = 0.0
    for M, N, K, name in [(264, 4096, 1536, "qkv bs8"), (264, 9216, 1536, "gate_up bs8"), (396, 1536, 2048, "o bs12"),
                          (396, 1536, 4608, "down bs12"), (1452, 1536, 4608, "down bs44"), (1452, 1536, 2048, "o bs44"),
                          (1056, 9216, 1536, "gate_up bs32"), (2112, 9216, 1536, "gate_up bs64"),
                          (3168, 9216, 1536, "gate_up bs96"), (12000, 9216, 1536, "gate_up prefill")]:
        a = (torch.randn(M, K, device="cuda") * 0.6).to(torch.float8_e4m3fn)
        b = (torch.randn(N, K, device="cuda") * 0.6).to(torch.float8_e4m3fn).t()
        sa = torch.rand(M, 1, device="cuda") * 0.02 + 1e-3
        sb = torch.rand(N, 1, device="cuda") * 0.02 + 1e-3
        same = torch.equal(fp8_scaled_mm(a, b, sa, sb, torch.bfloat16, None), G.dispatch(a, b, sa, sb, torch.bfloat16, None))
        t0 = bench(lambda: fp8_scaled_mm(a, b, sa, sb, torch.bfloat16, None))
        t1 = bench(lambda: G.dispatch(a, b, sa, sb, torch.bfloat16, None))
        sel = G._eligible(a, b, sa, sb, torch.bfloat16, None)
        print(f"  {name:16s} M={M:5d} N={N:5d} K={K:5d} {'triton' if sel else 'cutlass'} bitwise={same} "
              f"cutlass {t0:7.1f}us dispatch {t1:7.1f}us", flush=True)


if __name__ == "__main__":
    main()
