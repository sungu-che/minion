#!/usr/bin/env python3
"""Which rounding does SGLang's _triton_mrope_forward_fused perform on bf16 inputs? (diagnostic for fast_mrope)

Compares the Triton kernel's output with emulations:
  A  per-op bf16 rounding        : bf(bf(e*c) - bf(o*s))
  B  fp32 math, one final rounding: bf(e*c - o*s)              (fp32 products and difference)
  C  fma, first product fused     : bf(fma(e, c, -(o*s)))      (fp32)
  D  fma, second product fused    : bf(fma(-o, s, e*c))        (fp32)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    from sglang.srt.layers.rotary_embedding.triton_kernels import triton_mrope_fused

    from gravity_ocr import fast_mrope as F

    torch.manual_seed(3)
    hd, nqh, nkh, sec = 128, 16, 8, [16, 24, 24]
    max_pos = 32768
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    fr = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv)
    cache = torch.cat([fr.cos(), fr.sin()], -1).to(torch.bfloat16).cuda()
    t = 3168
    qkv = (torch.randn(t, (nqh + 2 * nkh) * hd, device="cuda") * 2).to(torch.bfloat16)
    p = torch.randint(100, 20000, (t,), device="cuda")
    pos = torch.stack([p, p, p])
    x1 = qkv.clone()
    triton_mrope_fused(x1[:, : nqh * hd], x1[:, nqh * hd: (nqh + nkh) * hd], cache, pos, sec, hd, hd, False, False, False, None)
    x2 = qkv.clone()
    F.mrope_inplace(x2[:, : nqh * hd], x2[:, nqh * hd: (nqh + nkh) * hd], cache, pos, sec, hd, hd)

    q = qkv[:, : nqh * hd].view(t, nqh, hd // 2, 2).float()
    e, o = q[..., 0], q[..., 1]
    cs = cache[p].float()
    c = cs[:, None, : hd // 2]
    s = cs[:, None, hd // 2:]
    bf = lambda z: z.to(torch.bfloat16).float()  # noqa: E731
    variants = {
        "A per-op bf16": (bf(bf(e * c) - bf(o * s)), bf(bf(o * c) + bf(e * s))),
        "B fp32 one rounding": (bf(e * c - o * s), bf(o * c + e * s)),
        "C fma(e,c,-(o*s))": (bf(torch.addcmul(-(o * s), e, c)), bf(torch.addcmul(e * s, o, c))),
        "D fma(-o,s,e*c)": (bf(torch.addcmul(e * c, -o, s)), bf(torch.addcmul(o * c, e, s))),
    }
    for name, xx in (("triton(sglang)", x1), ("fast_mrope", x2)):
        out = xx[:, : nqh * hd].view(t, nqh, hd // 2, 2).float()
        for vn, (ve, vo) in variants.items():
            n = int((out[..., 0] != ve).sum() + (out[..., 1] != vo).sum())
            print(f"{name:15s} vs {vn:22s}: {n} mismatching elements of {2 * ve.numel()}")
    d = (x1.float() - x2.float()).abs()
    print(f"triton vs fast: {int((x1 != x2).sum())} elements differ, max |diff| {float(d.max()):.4g}")


if __name__ == "__main__":
    main()
