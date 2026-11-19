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
_NORM = _ON and os.environ.get("TRACK2_FUSED_NORM", "1") == "1"
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


def _install_norm(C) -> int:
    """(residual add +) RMSNorm -> FP8 quant fused into the layer forwards (kernel == flashinfer norm + sgl-kernel quant,
    bitwise), feeding the same fp8_scaled_mm calls apply_fp8_linear would make:

      LM  (Glm4DecoderLayer): input_layernorm(+add) -> qkv_proj ; post_self_attn_layernorm -> add -> post_attention_layernorm
          -> gate_up_proj ; silu*up -> down_proj (fused as in the MLP patch); post_mlp_layernorm stays SGLang's.
      ViT (GlmOcrVisionBlock): norm1 -> qkv_proj (handed to the patched VisionAttention as _t2_pre_qkv); add + norm2 ->
          gate_up_proj ; silu*up -> down_proj; the final residual add stays a torch add.
    A layer whose linears are not on that exact FP8 path (or an unexpected shape) runs SGLang's forward."""
    from sglang.srt.layers.quantization import fp8_utils as fu
    from sglang.srt.models import glm4, glm_ocr

    scaled_mm = fu.fp8_scaled_mm

    def lin(layer, q8, s, dtype):
        return scaled_mm(q8, layer.weight, s, layer.weight_scale, out_dtype=dtype, bias=layer.bias)

    n = 0
    L = glm4.Glm4DecoderLayer
    if not getattr(L.forward, "_t2_fused", False):
        orig_l = L.forward

        def layer_forward(self, positions, hidden_states, forward_batch, residual):
            at, mlp = self.self_attn, self.mlp
            if (hidden_states.shape[0] == 0 or hidden_states.dtype != torch.bfloat16 or hidden_states.dim() != 2
                    or hidden_states.shape[-1] not in (1024, 1536) or not hidden_states.is_contiguous()
                    or (residual is not None and not residual.is_contiguous())
                    or not (_down_ok(at.qkv_proj) and _down_ok(mlp.gate_up_proj) and _down_ok(mlp.down_proj))):
                return orig_l(self, positions, hidden_states, forward_batch, residual)
            dt = hidden_states.dtype
            if residual is None:
                residual = hidden_states
                q8, s = C.norm_quant_fp8(hidden_states, self.input_layernorm.weight, self.input_layernorm.variance_epsilon,
                                         mode=C.NORM_MODE_PLAIN)
            else:
                # == fused_add_rmsnorm(hidden_states, residual): residual += hidden_states (in place); the normalized rows
                # go straight to FP8 (SGLang also overwrote hidden_states with them; nothing reads that buffer again)
                q8, s = C.norm_quant_fp8(hidden_states, self.input_layernorm.weight, self.input_layernorm.variance_epsilon,
                                         residual=residual, mode=C.NORM_MODE_ADD)
            qkv = lin(at.qkv_proj, q8, s, dt)
            q, k, v = qkv.split([at.q_size, at.kv_size, at.kv_size], dim=-1)
            q, k = at.rotary_emb(positions, q, k)
            o, _ = at.o_proj(at.attn(q, k, v, forward_batch))
            q8, s = C.norm_quant_fp8(o, self.post_attention_layernorm.weight, self.post_attention_layernorm.variance_epsilon,
                                     residual=residual, mode=C.NORM_MODE_PRE_ADD,
                                     weight0=self.post_self_attn_layernorm.weight,
                                     eps0=self.post_self_attn_layernorm.variance_epsilon)
            gate_up = lin(mlp.gate_up_proj, q8, s, dt)
            q8, s = C.silu_mul_quant_fp8(gate_up)
            down = lin(mlp.down_proj, q8, s, dt)
            return self.post_mlp_layernorm(down), residual

        layer_forward._t2_fused = True
        L.forward = layer_forward
        n += 1

    B = glm_ocr.GlmOcrVisionBlock
    if not getattr(B.forward, "_t2_fused", False) and os.environ.get("TRACK2_FUSED_NORM_VIT", "1") == "1":
        orig_b = B.forward

        def block_forward(self, x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin):
            at, mlp = self.attn, self.mlp
            if (x.dim() != 3 or x.shape[1] != 1 or x.shape[0] == 0 or x.dtype != torch.bfloat16
                    or x.shape[-1] not in (1024, 1536) or not x.is_contiguous()
                    or not getattr(at, "_track2_k2_inplace", False)
                    or not (_down_ok(at.qkv_proj) and _down_ok(mlp.gate_up_proj) and _down_ok(mlp.down_proj))):
                return orig_b(self, x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin)
            S, _, H = x.shape
            x2d = x.view(S, H)
            q8, s = C.norm_quant_fp8(x2d, self.norm1.weight, self.norm1.variance_epsilon, mode=C.NORM_MODE_PLAIN)
            at._t2_pre_qkv = lin(at.qkv_proj, q8, s, x.dtype).view(1, S, -1)   # consumed by the patched attention forward
            attn = at(x.view(1, S, H), cu_seqlens=cu_seqlens, rotary_pos_emb_cos=rotary_pos_emb_cos,
                      rotary_pos_emb_sin=rotary_pos_emb_sin)
            at.__dict__.pop("_t2_pre_qkv", None)
            attn2d = attn.reshape(S, H)
            if not attn2d.is_contiguous():
                attn2d = attn2d.contiguous()
            # == norm2(x2d, residual=attn2d): attn2d <- x + attn (x_after_add), normalized rows -> FP8
            q8, s = C.norm_quant_fp8(x2d, self.norm2.weight, self.norm2.variance_epsilon, residual=attn2d,
                                     mode=C.NORM_MODE_ADD)
            gate_up = lin(mlp.gate_up_proj, q8, s, x.dtype)
            q8, s = C.silu_mul_quant_fp8(gate_up)
            mlp_out = lin(mlp.down_proj, q8, s, x.dtype)
            return (attn2d + mlp_out).view(S, 1, H)

        block_forward._t2_fused = True
        B.forward = block_forward
        n += 1
    return n


def install() -> None:
    global _INSTALLED
    if _INSTALLED or not _ON:
        if not _ON:
            print("[track2-fused] OFF (TRACK2_FUSED=0)", flush=True)
        return
    _INSTALLED = True
    from gravity_ocr import cuda_sm89 as C

    parts = []
    try:  # must precede _install_mlp / _install_norm: they capture fp8_utils.fp8_scaled_mm
        from gravity_ocr import fp8_gemm

        parts.append(fp8_gemm.install())
    except Exception as e:
        parts.append(f"Triton FP8 GEMM unavailable ({e})")
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
    if _NORM and _MLP:
        if C.norm_available():
            try:
                parts.append(f"norm+fp8-quant layers ({_install_norm(C)} classes)")
            except Exception as e:
                parts.append(f"norm fusion unavailable ({e})")
        else:
            parts.append("norm fusion unavailable (libtrack2_cuda_sm89.so without track2_norm_quant_fp8)")
    print(f"[track2-fused] ON: {', '.join(parts) or 'nothing'}", flush=True)
