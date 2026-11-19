#!/usr/bin/env python3
"""Bitwise + timing checks of the v6 fused CUDA kernels against the kernels they replace (run inside track2-base).

  TRACK2_CUDA_SO=<built .so> /opt/sglang/.venv-sglang/bin/python tools/test_fused_kernels.py [k2] [smq] [mrope]

  k2    : vis_qk_norm_rope_inplace (K2 in place) == vis_qkv_sdpa_bf16 (K2) on q / k / v, and FA2-varlen / SDPA give the
          same outputs on the strided views as on K2's copies
  smq   : silu_mul_quant_fp8 == sglang_per_token_quant_fp8(sgl_kernel.silu_and_mul(x)) for every ACT / QNT arithmetic
          variant (mismatch counts per variant: the build's bitwise variant is the one with 0 / 0 everywhere)
  mrope : gravity_ocr.fast_mrope == SGLang's triton_mrope_fused (in place, bf16 cache), decode and prefill shapes
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gravity_ocr import cuda_sm89 as C  # noqa: E402


def bench(fn, n: int = 50) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n * 1000.0


def test_k2() -> bool:
    torch.manual_seed(0)
    h = 16
    all_ok = True
    for s, cdt in [(64, torch.float32), (1000, torch.float32), (16384, torch.float32), (4096, torch.bfloat16)]:
        qkv = (torch.randn(s, 3 * h * 64, device="cuda") * 2).to(torch.bfloat16)
        qn = (1 + 0.1 * torch.randn(64, device="cuda")).to(torch.bfloat16)
        kn = (1 + 0.1 * torch.randn(64, device="cuda")).to(torch.bfloat16)
        ang = torch.rand(s, 32, device="cuda") * 50
        cos = torch.cat([ang.cos(), ang.cos()], -1).to(cdt).contiguous()
        sin = torch.cat([ang.sin(), ang.sin()], -1).to(cdt).contiguous()
        q3, k3, v3 = C.vis_qkv_sdpa_bf16(qkv, qn, kn, cos, sin, 1e-6, h, True)
        x = qkv.clone()
        q, k, v = C.vis_qk_norm_rope_inplace(x, qn, kn, cos, sin, 1e-6, h)
        ok = torch.equal(q, q3) and torch.equal(k, k3) and torch.equal(v, v3)
        # several packed images: FA2-varlen on the strided views == on K2's contiguous copies
        bounds = sorted({0, s, int(s * 0.3), int(s * 0.55), int(s * 0.8)})
        cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
        mx = int(max(b - a for a, b in zip(bounds, bounds[1:])))
        o_old = torch.ops.aten._flash_attention_forward(q3, k3, v3, cu, cu, mx, mx, 0.0, False, False)[0]
        o_new = torch.ops.aten._flash_attention_forward(q, k, v, cu, cu, mx, mx, 0.0, False, False)[0]
        ok_fa2 = torch.equal(o_old, o_new)
        # one image: SDPA on [1, H, S, 64] views == on K2's [1, H, S, 64] copies
        q4, k4, v4 = C.vis_qkv_sdpa_bf16(qkv, qn, kn, cos, sin, 1e-6, h, False)
        s_old = torch.nn.functional.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
        s_new = torch.nn.functional.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                                                                 v.transpose(0, 1).unsqueeze(0), is_causal=False)
        ok_sdpa = torch.equal(s_old, s_new)
        t_old = bench(lambda: C.vis_qkv_sdpa_bf16(qkv, qn, kn, cos, sin, 1e-6, h, True))
        xb = qkv.clone()
        t_new = bench(lambda: C.vis_qk_norm_rope_inplace(xb, qn, kn, cos, sin, 1e-6, h))  # timing only (re-applied in place)
        all_ok &= ok and ok_fa2 and ok_sdpa
        print(f"K2 S={s:6d} cos={str(cdt)[6:]:8s} bitwise q/k/v={ok} fa2(views)==fa2(copies)={ok_fa2} "
              f"sdpa(views)==sdpa(copies)={ok_sdpa} | old {t_old:7.1f}us new {t_new:7.1f}us", flush=True)
    return all_ok


def test_smq() -> bool:
    import sgl_kernel
    from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_quant_fp8

    def ref(gu):
        y = torch.empty(gu.shape[0], gu.shape[1] // 2, dtype=gu.dtype, device=gu.device)
        sgl_kernel.silu_and_mul(gu, y)
        return y, sglang_per_token_quant_fp8(y)

    torch.manual_seed(1)
    best = {}
    for inter in (4608, 4096):
        for m in (1, 7, 33, 264, 3168, 16384):
            gu = (torch.randn(m, 2 * inter, device="cuda") * 1.5).to(torch.bfloat16)
            gu[: max(1, m // 8), : inter] *= 6  # a few large rows: other scale ranges
            y, (q_ref, s_ref) = ref(gu)
            res = []
            for a in (0, 1):
                for qm in (0, 1, 2):
                    q, s = C.silu_mul_quant_fp8(gu, a, qm)
                    nq = int((q.view(torch.uint8) != q_ref.view(torch.uint8)).sum())
                    ns = int((s.view(-1) != s_ref.view(-1)).sum())
                    res.append(f"a{a}q{qm}:{nq}/{ns}")
                    best[(a, qm)] = best.get((a, qm), 0) + nq + ns
            rq = []
            for qm in (0, 1, 2):
                q2, s2 = C.quant_fp8_rows(y, qm)
                rq.append(f"q{qm}:{int((q2.view(torch.uint8) != q_ref.view(torch.uint8)).sum())}/"
                          f"{int((s2.view(-1) != s_ref.view(-1)).sum())}")
            t_ref = bench(lambda: ref(gu))
            t_new = bench(lambda: C.silu_mul_quant_fp8(gu))
            print(f"SMQ I={inter} M={m:6d} q/scale mismatches {' '.join(res)} | quant-only {' '.join(rq)} | "
                  f"ref {t_ref:7.1f}us fused {t_new:7.1f}us", flush=True)
    winner = min(best, key=best.get)
    print(f"SMQ variant totals {best} -> bitwise variant ACT={winner[0]} QNT={winner[1]} "
          f"({'exact' if best[winner] == 0 else 'NOT exact'})", flush=True)
    return best[winner] == 0


def test_mrope() -> bool:
    from sglang.srt.layers.rotary_embedding.triton_kernels import triton_mrope_fused

    from gravity_ocr import fast_mrope as F

    torch.manual_seed(2)
    hd, nqh, nkh, sec = 128, 16, 8, [16, 24, 24]
    max_pos = 32768
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    fr = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv)
    cache = torch.cat([fr.cos(), fr.sin()], -1).to(torch.bfloat16).cuda()
    all_ok = True
    for t, kind in [(33, "decode-1"), (264, "decode-8"), (3168, "decode-96"), (16384, "prefill"), (1000, "prefill-small")]:
        qkv = (torch.randn(t, (nqh + 2 * nkh) * hd, device="cuda") * 2).to(torch.bfloat16)
        if kind.startswith("decode"):
            p = torch.randint(100, 20000, (t,), device="cuda")
            pos = torch.stack([p, p, p])
        else:
            pos = torch.stack([torch.randint(0, 3000, (t,), device="cuda") for _ in range(3)])
        q_a = qkv[:, : nqh * hd]
        k_a = qkv[:, nqh * hd: (nqh + nkh) * hd]
        x1 = qkv.clone()
        x2 = qkv.clone()
        triton_mrope_fused(x1[:, : nqh * hd], x1[:, nqh * hd: (nqh + nkh) * hd], cache, pos, sec, hd, hd, False, False,
                           False, None)
        F.mrope_inplace(x2[:, : nqh * hd], x2[:, nqh * hd: (nqh + nkh) * hd], cache, pos, sec, hd, hd)
        ok = torch.equal(x1, x2)
        all_ok &= ok
        y1 = qkv.clone()
        y2 = qkv.clone()
        t_old = bench(lambda: triton_mrope_fused(y1[:, : nqh * hd], y1[:, nqh * hd: (nqh + nkh) * hd], cache, pos, sec, hd, hd,
                                                 False, False, False, None))
        cfgs = []
        for bt, nw in ((1, 4), (2, 4), (4, 4), (4, 8), (8, 8)):
            x3 = qkv.clone()
            F.mrope_inplace(x3[:, : nqh * hd], x3[:, nqh * hd: (nqh + nkh) * hd], cache, pos, sec, hd, hd, bt, nw)
            same = torch.equal(x1, x3)
            all_ok &= same
            tt = bench(lambda: F.mrope_inplace(y2[:, : nqh * hd], y2[:, nqh * hd: (nqh + nkh) * hd], cache, pos, sec, hd, hd,
                                               bt, nw))
            cfgs.append(f"bt{bt}/w{nw}:{tt:6.1f}us{'' if same else '(DIFF)'}")
        del q_a, k_a
        print(f"MROPE T={t:6d} {kind:13s} bitwise={ok} | triton {t_old:7.1f}us | new {' '.join(cfgs)}", flush=True)
    return all_ok


if __name__ == "__main__":
    which = sys.argv[1:] or ["k2", "smq"]
    print(f"[test] lib={C._find_lib()} fused={C.fused_available()}", flush=True)
    ok = True
    if "k2" in which:
        ok &= test_k2()
    if "smq" in which:
        ok &= test_smq()
    if "mrope" in which:
        ok &= test_mrope()
    print("[test] ALL BITWISE OK" if ok else "[test] MISMATCHES (see above)", flush=True)
