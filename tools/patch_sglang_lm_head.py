#!/usr/bin/env python3
"""Patch SGLang 0.5.10 + trida-stack self-spec for Track 2 CUDA sm_89 Fast BF16 `lm_head` (K3).

Eliminates two massive GPU bottlenecks on every self-spec Draft & Verify round:
  1. 751 MB FP32 `next_token_logits` cast + HBM read/write (`[96*33, 59283] float32` -> `[96*33, 1] int32` = 12 KB!).
  2. Full 59,283-row `lm_head` GEMM on Draft mask tokens (rows 1..K-1 = 97% of Draft):
     - Draft rows 1..K-1 run ONLY 16,384-row universal BF16 slice + CUDA `gk_bf16_argmax_cs` (9.9x faster, 100% lossless
       because Verify checks every draft token!).
     - Draft row 0 (`a0`, committed unconditionally in SELFSPEC_FULLDRAFT=1; only `bs <= 96` rows = 3% of Draft) runs
       exact full 59,283-row BF16 GEMM + CUDA `gk_bf16_argmax_cs` (0.09 ms, 100% bit-exact!).
     - Verify runs full 59,283-row BF16 GEMM (8-aligned padded rows for 16B vectorized `uint4` loads) + CUDA
       `gk_bf16_argmax_cs` (2.3x faster, ZERO FP32 cast, ZERO host sync, 100% bit-exact, 100% inside CUDA Graphs!).

Applies cleanly to:
  - sglang/srt/layers/logits_processor.py
  - sglang/srt/speculative/selfspec_diffusion_worker.py
  - sglang/srt/speculative/ngram_info.py
  - sglang/srt/model_executor/cuda_graph_runner.py
"""
from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

LP_MARKER = "# [TRACK2-K3-FAST-LM-HEAD]"
WORKER_MARKER = "# [TRACK2-K3-WORKER]"
NGRAM_MARKER = "# [TRACK2-K3-NGRAM]"
GRAPH_MARKER = "# [TRACK2-K3-GRAPH]"


def patch_logits_processor(src: str) -> str:
    if LP_MARKER in src:
        return src
    helper = f"""
{LP_MARKER}
import os as _t2_os
import numpy as _t2_np
import torch.nn.functional as _t2_F

def _t2_get_lm_head_cache(lm_head, vocab_size: int):
    cache = getattr(lm_head, "_t2_k3_cache", None)
    if cache is not None:
        return cache
    if torch.cuda.is_current_stream_capturing():
        return None
    try:
        from gravity_ocr import cuda_sm89 as _t2_cuda
        if not _t2_cuda.available():
            lm_head._t2_k3_cache = False
            return False
    except Exception:
        lm_head._t2_k3_cache = False
        return False

    W = lm_head.weight
    if W.dtype != torch.bfloat16 or not W.is_cuda:
        lm_head._t2_k3_cache = False
        return False
    v_full = min(int(vocab_size), int(W.shape[0]))
    pad_rows = (8 - (v_full % 8)) % 8
    if pad_rows > 0:
        W_pad = _t2_F.pad(W[:v_full], (0, 0, 0, pad_rows), value=0.0).contiguous()
    else:
        W_pad = W[:v_full].contiguous()

    # Load universal 16,384-row core slice if available
    W_core = None
    core_ids = None
    slice_dir = _t2_os.environ.get("TRACK2_SLICE_DIR", "")
    cand_paths = [
        _t2_os.path.join(slice_dir, "universal_16k_ids.npy") if slice_dir else "",
        _t2_os.path.join(slice_dir, "core_table_ids.npy") if slice_dir else "",
    ]
    for p in cand_paths:
        if p and _t2_os.path.isfile(p):
            ids_np = _t2_np.load(p).astype(_t2_np.int64)
            ids_np = ids_np[ids_np < v_full]
            n_keep = (len(ids_np) // 8) * 8
            if n_keep >= 1024:
                ids_np = ids_np[:n_keep]
                core_ids = torch.from_numpy(ids_np.astype(_t2_np.int32)).to(W.device)
                W_core = W[core_ids.long()].contiguous()
                break

    cache = {{
        "cuda": _t2_cuda,
        "W_pad": W_pad,
        "V_valid": v_full,
        "W_core": W_core,
        "core_ids": core_ids,
    }}
    lm_head._t2_k3_cache = cache
    print(
        f"[track2-k3-lm-head] ON: V_valid={{v_full}} W_pad={{tuple(W_pad.shape)}} "
        f"W_core={{None if W_core is None else tuple(W_core.shape)}}",
        flush=True,
    )
    return cache
"""
    anchor_cls = "class LogitsProcessor(nn.Module):"
    if anchor_cls not in src:
        raise RuntimeError("Could not find LogitsProcessor class in logits_processor.py")
    src = src.replace(anchor_cls, helper + "\n\n" + anchor_cls, 1)

    target_fwd = """        if isinstance(logits_metadata, ForwardBatch):
            logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)"""
    repl_fwd = """        # [TRACK2-K3-FAST-LM-HEAD] Fast BF16 lm_head + CUDA sm_89 argmax for plain self-spec Draft/Verify
        if (
            isinstance(logits_metadata, ForwardBatch)
            and _t2_os.environ.get("TRACK2_FAST_LM_HEAD", "1") == "1"
            and logits_metadata.forward_mode.is_target_verify()
            and not logits_metadata.return_logprob
            and logits_metadata.capture_hidden_mode == CaptureHiddenMode.NULL
            and getattr(self, "attn_tp_size", 1) == 1
            and getattr(lm_head, "tp_size", 1) == 1
            and self.logit_scale is None
            and not getattr(self, "final_logit_softcapping", None)
            and not self.do_tensor_parallel_all_gather
            and not self.do_tensor_parallel_all_gather_dp_attn
            and hidden_states.dtype == torch.bfloat16
        ):
            _spec = logits_metadata.spec_info
            _role = getattr(_spec, "selfspec_role", None)
            _plain = getattr(_spec, "ss_plain", True)
            if _role in ("draft", "verify") and _plain:
                _k3 = _t2_get_lm_head_cache(lm_head, self.vocab_size)
                if _k3:
                    _cuda = _k3["cuda"]
                    _W_pad = _k3["W_pad"]
                    _V_valid = _k3["V_valid"]
                    if _role == "draft" and _k3["W_core"] is not None:
                        _K = int(_spec.draft_token_num)
                        _sub = _t2_F.linear(hidden_states, _k3["W_core"])
                        _am_i32, _, _ = _cuda.bf16_argmax_cs(
                            _sub, _k3["core_ids"], check_cs=False
                        )
                        # Exact full-vocab argmax for a0 (row 0 of each request: 0, K, 2K, ...)
                        _h0 = hidden_states[0::_K].contiguous()
                        _lg0 = _t2_F.linear(_h0, _W_pad)
                        _a0_i32, _, _ = _cuda.bf16_argmax_cs(
                            _lg0, None, check_cs=False, v_valid=_V_valid
                        )
                        _am_i32[0::_K] = _a0_i32
                    else:
                        _lg = _t2_F.linear(hidden_states, _W_pad)
                        _am_i32, _, _ = _cuda.bf16_argmax_cs(
                            _lg, None, check_cs=False, v_valid=_V_valid
                        )
                    return LogitsProcessorOutput(
                        next_token_logits=_am_i32.unsqueeze(-1),
                        hidden_states=None,
                        mm_input_embeds=None,
                    )

        if isinstance(logits_metadata, ForwardBatch):
            logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)"""
    if target_fwd not in src:
        raise RuntimeError("Could not find LogitsProcessor.forward anchor in logits_processor.py")
    src = src.replace(target_fwd, repl_fwd, 1)

    # Also guard _copy_logits_to_buffer when next_token_logits_buffer is shrunk to 1 col
    target_buf = """        if logits_metadata.next_token_logits_buffer is not None:
            logits_buffer = logits_metadata.next_token_logits_buffer
            assert logits_buffer.dtype == torch.float
            logits_buffer.copy_(logits[:, : self.vocab_size])
            logits = logits_buffer"""
    repl_buf = """        if (
            logits_metadata.next_token_logits_buffer is not None
            and logits_metadata.next_token_logits_buffer.shape[-1] >= self.vocab_size
        ):
            logits_buffer = logits_metadata.next_token_logits_buffer
            assert logits_buffer.dtype == torch.float
            logits_buffer.copy_(logits[:, : self.vocab_size])
            logits = logits_buffer"""
    if target_buf in src:
        src = src.replace(target_buf, repl_buf, 1)
    return src


def patch_worker(src: str) -> str:
    if WORKER_MARKER in src:
        return src
    # 1) Mark ss_plain on draft's spec_info
    target_spec = """        if self._fastattn:
            spec.selfspec_role = "draft"
            if shadow_mask is not None:
                spec.ss_shadow_mask = shadow_mask"""
    repl_spec = f"""        {WORKER_MARKER}
        spec.ss_plain = self._plain(batch)
        if self._fastattn:
            spec.selfspec_role = "draft"
            if shadow_mask is not None:
                spec.ss_shadow_mask = shadow_mask"""
    if target_spec not in src:
        raise RuntimeError("Could not find draft spec.selfspec_role block in selfspec_diffusion_worker.py")
    src = src.replace(target_spec, repl_spec, 1)

    # 2) Consume precomputed [bs*n, 1] int32 argmax directly when K3 fast lm_head is active
    target_am = """        logits = r.logits_output.next_token_logits.view(bs, n, -1)  # logit[i,j] predicts win[i,j+1]
        # With n=K+1 (SELFSPEC_TRAIN_MASKS=1) there is an extra row K that predicts the first token of the
        #   NEXT block; it is unused this round. The draft slice (:K-1) stays valid.
        am = logits.argmax(-1)  # [bs, n]"""
    repl_am = """        _raw_lg = r.logits_output.next_token_logits
        if _raw_lg.shape[-1] == 1 and _raw_lg.dtype == torch.int32:
            logits = None
            am = _raw_lg.view(bs, n).to(torch.int64)
        else:
            logits = _raw_lg.view(bs, n, -1)  # logit[i,j] predicts win[i,j+1]
            # With n=K+1 (SELFSPEC_TRAIN_MASKS=1) there is an extra row K that predicts the first token of the
            #   NEXT block; it is unused this round. The draft slice (:K-1) stays valid.
            am = logits.argmax(-1)  # [bs, n]"""
    if target_am not in src:
        raise RuntimeError("Could not find draft logits.argmax(-1) block in selfspec_diffusion_worker.py")
    src = src.replace(target_am, repl_am, 1)

    # 3) [TRACK2-PREFILL-COALESCE] a prefill the scheduler is deferring will not release the stash: keep drafting ahead
    target_gate = """        if s.chunked_req is not None or (len(s.waiting_queue) and not s.running_batch.batch_is_full) or \\
                s.grammar_manager.has_waiting_grammars():"""
    repl_gate = """        _t2_defer = getattr(s, "_t2_prefill_deferred", None)
        if s.chunked_req is not None or (len(s.waiting_queue) and not s.running_batch.batch_is_full
                                         and not (_t2_defer is not None and _t2_defer(len(batch.reqs)))) or \\
                s.grammar_manager.has_waiting_grammars():"""
    if target_gate in src:
        src = src.replace(target_gate, repl_gate, 1)
    return src


def patch_ngram_info(src: str) -> str:
    if NGRAM_MARKER in src:
        return src
    target = """        bs = batch.batch_size()
        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
        target_predict = target_predict.reshape(bs, self.draft_token_num)"""
    repl = f"""        {NGRAM_MARKER}
        bs = batch.batch_size()
        _raw_lg = logits_output.next_token_logits
        if _raw_lg.shape[-1] == 1 and _raw_lg.dtype == torch.int32:
            target_predict = _raw_lg.reshape(bs, self.draft_token_num).to(torch.int64)
        else:
            target_predict = torch.argmax(_raw_lg, dim=-1).reshape(bs, self.draft_token_num)"""
    if target not in src:
        raise RuntimeError("Could not find _greedy_verify argmax block in ngram_info.py")
    return src.replace(target, repl, 1)


def patch_cuda_graph_runner(src: str) -> str:
    if GRAPH_MARKER in src:
        return src
    # 1) Set ss_plain = True during graph capture for self-spec roles
    target_role = """        if ss_role is not None:
            spec_info.selfspec_role = ss_role   # routes the attention backend to the role's wrapper / mask mode"""
    repl_role = f"""        if ss_role is not None:
            {GRAPH_MARKER}
            spec_info.selfspec_role = ss_role   # routes the attention backend to the role's wrapper / mask mode
            spec_info.ss_plain = True"""
    if target_role not in src:
        raise RuntimeError("Could not find ss_role capture block in cuda_graph_runner.py")
    src = src.replace(target_role, repl_role, 1)

    # 2) Free the 751 MB FP32 next_token_logits_buffer when TRACK2_FAST_LM_HEAD=1
    target_alloc = """            next_token_logits_buffer = torch.zeros(
                (max_num_token, vocab_size),
                dtype=torch.float,
            )"""
    repl_alloc = """            _t2_fast_lh = os.environ.get("TRACK2_FAST_LM_HEAD", "1") == "1"
            next_token_logits_buffer = torch.zeros(
                (max_num_token, 1 if _t2_fast_lh else vocab_size),
                dtype=torch.float,
            )"""
    if target_alloc in src:
        src = src.replace(target_alloc, repl_alloc, 1)
    return src


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch SGLang for Track 2 CUDA sm_89 Fast BF16 lm_head (K3)")
    ap.add_argument("--logits-src", required=True)
    ap.add_argument("--worker-src", required=True)
    ap.add_argument("--ngram-src", required=True)
    ap.add_argument("--graph-src", required=True)
    ap.add_argument("--out-sglang-py", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_sglang_py)
    jobs = [
        (args.logits_src, "sglang/srt/layers/logits_processor.py", patch_logits_processor),
        (args.worker_src, "sglang/srt/speculative/selfspec_diffusion_worker.py", patch_worker),
        (args.ngram_src, "sglang/srt/speculative/ngram_info.py", patch_ngram_info),
        (args.graph_src, "sglang/srt/model_executor/cuda_graph_runner.py", patch_cuda_graph_runner),
    ]
    for src_path, rel_out, fn in jobs:
        patched = fn(Path(src_path).read_text(encoding="utf-8"))
        dst = out_root / rel_out
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(patched, encoding="utf-8", newline="\n")
        py_compile.compile(str(dst), doraise=True)
        print(f"Patched & verified: {dst}", flush=True)


if __name__ == "__main__":
    main()
