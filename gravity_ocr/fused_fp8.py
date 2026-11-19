"""v6 fused kernels wired into SGLang's GLM-OCR forward (installed from models/glm_ocr.py load_weights, i.e. before any
CUDA graph is captured).

  MLP   (Glm4MLP: language model, Glm4vVisionMLP: ViT)
        gate_up -> [silu_and_mul -> sglang_per_token_quant_fp8] -> fp8_scaled_mm(down_proj)
        becomes gate_up -> [cuda_sm89.silu_mul_quant_fp8] -> the same fp8_scaled_mm call apply_fp8_linear makes.
        The fused kernel's (q, scale) are bitwise equal to the two kernels it replaces (tools/test_fused_kernels.py smq),
        so the down_proj output is unchanged.
  MROPE (MRotaryEmbedding.forward_triton) -> gravity_ocr.fast_mrope (bitwise equal, tools/test_fused_kernels.py mrope).
  K2    in-place ViT q/k norm + RoPE: lives in the vision.py patch (tools/patch_sglang_vit_l40s.py, TRACK2_K2_INPLACE).

Switches (default on): TRACK2_FUSED=0 turns all of them off; TRACK2_FUSED_MLP / TRACK2_FUSED_MROPE individually.
A layer that does not match the exact FP8 path the fused code reproduces (per-channel weight scale, dynamic per-token
activations, CUTLASS fp8_scaled_mm, tp 1) keeps SGLang's own forward.
"""
from __future__ import annotations

import os

import torch

_ON = os.environ.get("TRACK2_FUSED", "1") == "1"
_MLP = _ON and os.environ.get("TRACK2_FUSED_MLP", "1") == "1"
_MROPE = _ON and os.environ.get("TRACK2_FUSED_MROPE", "1") == "1"
_INSTALLED = False


def _down_ok(layer) -> bool:
    """True when `layer` (a RowParallelLinear) would run apply_fp8_linear's CUTLASS per-token path."""
    ok = getattr(layer, "_t2_fused_ok", None)
    if ok is not None:
        return ok
    w = getattr(layer, "weight", None)
    if w is None or w.dtype != torch.float8_e4m3fn:
        return False  # not quantized (yet): decide again on the next call, do not cache
    try:
        from sglang.srt.layers.quantization import fp8_utils as fu
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        qm = layer.quant_method
        ws = layer.weight_scale
        ok = bool(
            isinstance(qm, Fp8LinearMethod)
            and not getattr(qm, "use_marlin", False)
            and not getattr(qm, "use_mxfp8", False)
            and not getattr(qm, "block_quant", False)
            and getattr(layer, "input_scale", None) is None
            and getattr(qm, "cutlass_fp8_supported", False)
            and ws.numel() == w.shape[1]
            and w.shape[0] % 16 == 0
            and w.shape[1] % 16 == 0
            and not getattr(fu, "use_triton_w8a8_fp8_kernel", False)
            and getattr(fu, "_is_cuda", False)
            and int(getattr(layer, "tp_size", 1)) == 1
            and not getattr(layer, "skip_bias_add", False)
        )
    except Exception:
        ok = False
    layer._t2_fused_ok = ok
    return ok


def _install_mlp(C) -> int:
    from sglang.srt.layers.quantization import fp8_utils as fu

    scaled_mm = fu.fp8_scaled_mm
    n = 0

    def make(orig):
        def forward(self, x, *args, **kwargs):
            dp = self.down_proj
            if not _down_ok(dp):
                return orig(self, x, *args, **kwargs)
            gate_up, _ = self.gate_up_proj(x)
            if gate_up.shape[-1] != 2 * dp.weight.shape[0]:  # unexpected layout: SGLang's ops on this gate_up
                h, _ = dp(self.act_fn(gate_up))
                return h
            q, s = C.silu_mul_quant_fp8(gate_up)
            # == apply_fp8_linear's CUTLASS branch: fp8_scaled_mm(qinput, weight, x_scale, weight_scale, out_dtype, bias)
            out = scaled_mm(q, dp.weight, s, dp.weight_scale, out_dtype=gate_up.dtype, bias=dp.bias)
            return out.view(*gate_up.shape[:-1], out.shape[-1])

        forward._t2_fused = True
        forward._t2_orig = orig
        return forward

    from sglang.srt.models import glm4, glm4v

    for cls in (glm4.Glm4MLP, glm4v.Glm4vVisionMLP):
        if not getattr(cls.forward, "_t2_fused", False):
            cls.forward = make(cls.forward)
            n += 1
    return n


def _install_mrope() -> bool:
    from sglang.srt.layers.rotary_embedding import mrope as M

    from gravity_ocr import fast_mrope as F

    cls = M.MRotaryEmbedding
    if getattr(cls.forward_triton, "_t2_fused", False):
        return True
    orig = cls.forward_triton

    def forward_triton(self, positions, query, key):
        self._match_cos_sin_cache_dtype(query)
        if F.supported(query, key, self.cos_sin_cache, positions, self.mrope_section, self.head_size, self.rotary_dim,
                       self.mrope_interleaved, self.mrope_interleaved_glm, self.is_neox_style):
            F.mrope_inplace(query, key, self.cos_sin_cache, positions, self.mrope_section, self.head_size, self.rotary_dim)
            return query, key
        return orig(self, positions, query, key)

    forward_triton._t2_fused = True
    cls.forward_triton = forward_triton
    return True


def install() -> None:
    global _INSTALLED
    if _INSTALLED or not _ON:
        if not _ON:
            print("[track2-fused] OFF (TRACK2_FUSED=0)", flush=True)
        return
    _INSTALLED = True
    from gravity_ocr import cuda_sm89 as C

    parts = []
    if _MLP:
        if C.fused_available():
            parts.append(f"silu*up+fp8-quant MLP ({_install_mlp(C)} classes)")
        else:
            parts.append("MLP fusion unavailable (libtrack2_cuda_sm89.so without fused kernels)")
    if _MROPE:
        try:
            _install_mrope()
            parts.append("fast M-RoPE")
        except Exception as e:  # triton / signature drift: keep SGLang's kernel
            parts.append(f"M-RoPE fusion unavailable ({e})")
    print(f"[track2-fused] ON: {', '.join(parts) or 'nothing'}", flush=True)
