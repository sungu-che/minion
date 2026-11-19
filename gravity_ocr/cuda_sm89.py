"""Zero-copy ctypes wrapper for Track 2 native BF16 CUDA (sm_89) kernels.

Loads `gravity_ocr/_native/libtrack2_cuda_sm89.so` and dispatches directly on
`torch.cuda.current_stream().cuda_stream` so every call is 100% CUDA Graph capture safe.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional, Tuple

import torch

_LIB: Optional[ctypes.CDLL] = None
_TRIED = False

# Precomputed normalization constants matching Glm46VImageProcessorFast / gravity_ocr.preprocess:
#   out = (u8 / 255.0 - mean) / std = u8 * (1 / (255 * std)) + (-mean / std)
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)
_SCALE = tuple(1.0 / (255.0 * s) for s in _STD)
_BIAS = tuple(-m / s for m, s in zip(_MEAN, _STD))


def _find_lib() -> Optional[Path]:
    env_p = os.environ.get("TRACK2_CUDA_SO")
    if env_p and os.path.isfile(env_p):
        return Path(env_p)
    here = Path(__file__).resolve().parent / "_native" / "libtrack2_cuda_sm89.so"
    if here.is_file():
        return here
    return None


def load() -> Optional[ctypes.CDLL]:
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    path = _find_lib()
    if path is None:
        return None
    try:
        lib = ctypes.CDLL(str(path))
        lib.track2_u8_norm_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        lib.track2_u8_norm_bf16.restype = ctypes.c_int

        lib.track2_vis_qkv_sdpa_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.track2_vis_qkv_sdpa_bf16.restype = ctypes.c_int

        lib.track2_bf16_argmax_cs.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.track2_bf16_argmax_cs.restype = ctypes.c_int

        _LIB = lib
    except Exception:
        _LIB = None
    return _LIB


def available() -> bool:
    return load() is not None


def u8_norm_bf16(x_u8: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Normalize `[N, 588]` uint8 single-frame patches directly into `[N, 588]` BF16 on GPU."""
    lib = load()
    assert lib is not None
    x_u8 = x_u8.contiguous()
    n = int(x_u8.shape[0])
    if out is None:
        out = torch.empty((n, 588), dtype=torch.bfloat16, device=x_u8.device)
    stream = torch.cuda.current_stream(x_u8.device).cuda_stream
    err = lib.track2_u8_norm_bf16(
        x_u8.data_ptr(),
        out.data_ptr(),
        n,
        _SCALE[0], _BIAS[0],
        _SCALE[1], _BIAS[1],
        _SCALE[2], _BIAS[2],
        stream,
    )
    if err != 0:
        raise RuntimeError(f"track2_u8_norm_bf16 CUDA error {err}")
    return out


def vis_qkv_sdpa_bf16(
    qkv: torch.Tensor,
    qn_weight: torch.Tensor,
    kn_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    num_heads: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QKV split + Q/K head RMSNorm + NeoX 2D RoPE -> contiguous `[1, H, S, 64]` SDPA tensors."""
    lib = load()
    assert lib is not None
    qkv_c = qkv.contiguous()
    s = int(qkv_c.numel() // (3 * num_heads * 64))
    dev = qkv_c.device
    q_out = torch.empty((1, num_heads, s, 64), dtype=torch.bfloat16, device=dev)
    k_out = torch.empty((1, num_heads, s, 64), dtype=torch.bfloat16, device=dev)
    v_out = torch.empty((1, num_heads, s, 64), dtype=torch.bfloat16, device=dev)
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()
    cos_is_bf16 = 1 if cos_c.dtype == torch.bfloat16 else 0
    if not cos_is_bf16 and cos_c.dtype != torch.float32:
        cos_c = cos_c.float()
        sin_c = sin_c.float()
    stream = torch.cuda.current_stream(dev).cuda_stream
    err = lib.track2_vis_qkv_sdpa_bf16(
        qkv_c.data_ptr(),
        qn_weight.data_ptr(),
        kn_weight.data_ptr(),
        cos_c.data_ptr(),
        sin_c.data_ptr(),
        cos_is_bf16,
        float(eps),
        q_out.data_ptr(),
        k_out.data_ptr(),
        v_out.data_ptr(),
        s,
        num_heads,
        stream,
    )
    if err != 0:
        raise RuntimeError(f"track2_vis_qkv_sdpa_bf16 CUDA error {err}")
    return q_out, k_out, v_out


def bf16_argmax_cs(
    sub_logits: torch.Tensor,
    core_ids: Optional[torch.Tensor] = None,
    hidden: Optional[torch.Tensor] = None,
    mu_rest: Optional[torch.Tensor] = None,
    w_rest_bound: float = 0.0,
    check_cs: bool = False,
    v_valid: Optional[int] = None,
    out_id: Optional[torch.Tensor] = None,
    out_val: Optional[torch.Tensor] = None,
    out_rescue: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fused BF16 vectorized argmax + core_ids remap + optional Cauchy-Schwarz bound check."""
    lib = load()
    assert lib is not None
    sub_c = sub_logits.contiguous()
    m, v_stride = int(sub_c.shape[0]), int(sub_c.shape[1])
    vv = int(v_valid) if v_valid is not None else v_stride
    dev = sub_c.device
    if out_id is None:
        out_id = torch.empty((m,), dtype=torch.int32, device=dev)
    if check_cs:
        if out_val is None:
            out_val = torch.empty((m,), dtype=torch.float32, device=dev)
        if out_rescue is None:
            out_rescue = torch.empty((m,), dtype=torch.uint8, device=dev)
    h_dim = int(hidden.shape[1]) if hidden is not None else 0
    stream = torch.cuda.current_stream(dev).cuda_stream
    err = lib.track2_bf16_argmax_cs(
        sub_c.data_ptr(),
        core_ids.data_ptr() if core_ids is not None else 0,
        hidden.contiguous().data_ptr() if (check_cs and hidden is not None) else 0,
        mu_rest.data_ptr() if (check_cs and mu_rest is not None) else 0,
        float(w_rest_bound),
        out_id.data_ptr(),
        out_val.data_ptr() if out_val is not None else 0,
        out_rescue.data_ptr() if (check_cs and out_rescue is not None) else 0,
        m,
        v_stride,
        vv,
        h_dim,
        1 if check_cs else 0,
        stream,
    )
    if err != 0:
        raise RuntimeError(f"track2_bf16_argmax_cs CUDA error {err}")
    return out_id, out_val, out_rescue
