"""Faster in-place M-RoPE for the GLM-OCR language model (drop-in for SGLang's triton_mrope_fused, non-NeoX layout).

SGLang's _triton_mrope_forward_fused runs one program per token and reads the rotation pairs with stride-2 gathers
(even / odd elements). Here a program rotates BLOCK_T tokens, loads every head row contiguously (16-byte vectors) and
splits it into (even, odd) in registers with tl.split; the arithmetic is the same bf16 expressions as SGLang's kernel
(cos / sin rows assembled from the t / h / w sections by masked loads + adds, then `e * cos - o * sin`, `o * cos + e * sin`),
with SGLang's rounding pinned by explicit fma (its PTX: even = bf16(e*cos - bf16(o*sin)), odd = bf16(e*sin + bf16(o*cos))),
so the result is bitwise equal (tools/test_fused_kernels.py mrope).

Covers the configuration GLM-OCR uses: mrope_interleaved False, mrope_interleaved_glm False, is_neox_style False,
rotary_dim == head_size, power-of-two head counts / head size, cache dtype == q dtype. Anything else -> caller falls back.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

BLOCK_T = int(os.environ.get("TRACK2_MROPE_BLOCK_T", "4"))
NUM_WARPS = int(os.environ.get("TRACK2_MROPE_WARPS", "4"))


@triton.jit
def _t2_mrope_kernel(
    q_ptr, k_ptr, cs_ptr, pos_ptr,
    q_stride, k_stride, pos_stride, T,
    n_qh: tl.constexpr, n_kh: tl.constexpr, hd: tl.constexpr, rd: tl.constexpr,
    sec_t: tl.constexpr, sec_h: tl.constexpr, BT: tl.constexpr,
):
    pid = tl.program_id(0)
    toks = pid * BT + tl.arange(0, BT)
    tm = toks < T
    half: tl.constexpr = rd // 2
    pt = tl.load(pos_ptr + toks, mask=tm, other=0)
    ph = tl.load(pos_ptr + pos_stride + toks, mask=tm, other=0)
    pw = tl.load(pos_ptr + 2 * pos_stride + toks, mask=tm, other=0)
    co = tl.arange(0, half)
    t_m = co < sec_t
    h_m = (co >= sec_t) & (co < sec_t + sec_h)
    w_m = (co >= sec_t + sec_h) & (co < half)
    tmc = tm[:, None]
    t_cos = tl.load(cs_ptr + pt[:, None] * rd + co[None, :], mask=tmc & t_m[None, :], other=0)
    h_cos = tl.load(cs_ptr + ph[:, None] * rd + co[None, :], mask=tmc & h_m[None, :], other=0)
    w_cos = tl.load(cs_ptr + pw[:, None] * rd + co[None, :], mask=tmc & w_m[None, :], other=0)
    t_sin = tl.load(cs_ptr + pt[:, None] * rd + half + co[None, :], mask=tmc & t_m[None, :], other=0)
    h_sin = tl.load(cs_ptr + ph[:, None] * rd + half + co[None, :], mask=tmc & h_m[None, :], other=0)
    w_sin = tl.load(cs_ptr + pw[:, None] * rd + half + co[None, :], mask=tmc & w_m[None, :], other=0)
    cos_row = t_cos + h_cos + w_cos
    sin_row = t_sin + h_sin + w_sin
    c3 = cos_row[:, None, :]
    s3 = sin_row[:, None, :]
    d = tl.arange(0, hd)
    m3 = tm[:, None, None]

    qh = tl.arange(0, n_qh)
    qoff = toks[:, None, None].to(tl.int64) * q_stride + qh[None, :, None] * hd + d[None, None, :]
    xq = tl.load(q_ptr + qoff, mask=m3, other=0).to(sin_row.dtype)
    qe, qo = tl.split(tl.reshape(xq, (BT, n_qh, half, 2)))
    cq = tl.broadcast_to(c3, (BT, n_qh, half))
    sq = tl.broadcast_to(s3, (BT, n_qh, half))
    # SGLang's kernel compiles `e * cos - o * sin` / `o * cos + e * sin` (bf16) to native bf16 FMAs with the e-product fused:
    # even = bf16(e * cos - bf16(o * sin)), odd = bf16(e * sin + bf16(o * cos)); pinned here with explicit fma
    yq = tl.reshape(tl.join(tl.fma(qe, cq, -(qo * sq)), tl.fma(qe, sq, qo * cq)), (BT, n_qh, hd))
    tl.store(q_ptr + qoff, yq, mask=m3)

    kh = tl.arange(0, n_kh)
    koff = toks[:, None, None].to(tl.int64) * k_stride + kh[None, :, None] * hd + d[None, None, :]
    xk = tl.load(k_ptr + koff, mask=m3, other=0).to(sin_row.dtype)
    ke, ko = tl.split(tl.reshape(xk, (BT, n_kh, half, 2)))
    ck = tl.broadcast_to(c3, (BT, n_kh, half))
    sk = tl.broadcast_to(s3, (BT, n_kh, half))
    yk = tl.reshape(tl.join(tl.fma(ke, ck, -(ko * sk)), tl.fma(ke, sk, ko * ck)), (BT, n_kh, hd))
    tl.store(k_ptr + koff, yk, mask=m3)


def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def supported(q: torch.Tensor, k: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor, mrope_section, head_size: int,
              rotary_dim: int, interleaved: bool, interleaved_glm: bool, neox: bool) -> bool:
    if interleaved or interleaved_glm or neox or rotary_dim != head_size or positions.ndim != 2:
        return False
    if q.dtype != cache.dtype or k.dtype != cache.dtype or q.stride(-1) != 1 or k.stride(-1) != 1:
        return False
    if len(mrope_section) != 3 or sum(mrope_section) != rotary_dim // 2:
        return False
    n_qh, n_kh = q.shape[-1] // head_size, k.shape[-1] // head_size
    return _pow2(n_qh) and _pow2(n_kh) and _pow2(head_size) and cache.is_contiguous() and positions.stride(-1) == 1


def mrope_inplace(q: torch.Tensor, k: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor, mrope_section,
                  head_size: int, rotary_dim: int, block_t: int | None = None, num_warps: int | None = None) -> None:
    """q [T, n_qh * hd], k [T, n_kh * hd] (row-strided views allowed), cache [max_pos, rd] (cos | sin), positions [3, T]."""
    t = int(q.shape[0])
    if t == 0:
        return
    bt = block_t or BLOCK_T
    _t2_mrope_kernel[(triton.cdiv(t, bt),)](
        q, k, cache, positions,
        q.stride(0), k.stride(0), positions.stride(0), t,
        q.shape[-1] // head_size, k.shape[-1] // head_size, head_size, rotary_dim,
        int(mrope_section[0]), int(mrope_section[1]), bt,
        num_warps=num_warps or NUM_WARPS,
    )
