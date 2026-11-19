#!/usr/bin/env python3
"""Patch trida-stack's selfspec_fast_attn.py: ONE attention pass per layer for the self-spec DRAFT forward.

Why:
  The fast path runs every draft layer's attention twice: NON_CAUSAL (rows 1..K-1, the masks: prefix + seed + all
  masks) and then CAUSAL over the whole window only to copy its row 0 (the seed: prefix + seed). Both passes read the
  whole paged KV of every request; on L40S the extra pass is ~0.28 s of a 4.5 s run (2 of the 3 KV sweeps per round).

What (TRACK2_DRAFT_MASK=1, default):
  The draft wrappers get a packed custom mask (FlashInfer MaskMode CUSTOM): all ones, except row 0's columns
  [L + 1, L + K) (the K - 1 masks after the seed). One pass then gives every row its pattern: row 0 = prefix + seed
  (exactly the causal row 0, bottom-right aligned), rows 1..K-1 = everything. The mask and its byte offsets are written
  on the GPU by one Triton kernel next to the kv-indices kernel, into static buffers (CUDA-graph safe; the graphs read
  them at replay). FlashInfer's plan does not depend on the mask mode (same tiles), and masked keys add exact zeros.
"""
from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

MARKER = "# [TRACK2-DRAFT-MASK]"

KERNEL = '''

# [TRACK2-DRAFT-MASK] one-pass draft attention: packed custom mask written on the GPU
_T2_DRAFT_MASK = os.environ.get("TRACK2_DRAFT_MASK", "1") == "1"


@triton.jit
def _t2_draft_mask_kernel(
    seq_lens_ptr,
    mask_indptr_ptr,
    mask_ptr,
    K: tl.constexpr,
    BS_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """one program per sequence i (kv = L_i + K): bits of the row-major [K, L_i + K] mask packed little-endian into
    ceil(K * (L_i + K) / 8) bytes at mask_indptr[i]; all ones except row 0, columns [L_i + 1, L_i + K)."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BS_POW2)
    Ls = tl.load(seq_lens_ptr + offs, mask=offs < pid, other=0).to(tl.int64)
    seg = (K * (Ls + K) + 7) // 8
    prev = tl.sum(tl.where(offs < pid, seg, 0), axis=0)
    L = tl.load(seq_lens_ptr + pid).to(tl.int64)
    nbytes = (K * (L + K) + 7) // 8
    if pid == 0:
        tl.store(mask_indptr_ptr, 0)
    tl.store(mask_indptr_ptr + pid + 1, (prev + nbytes).to(tl.int32))
    for it in range(tl.cdiv(nbytes, BLOCK)):
        idx = it * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        lo = tl.minimum(tl.maximum(L + 1 - idx * 8, 0), 8)
        hi = tl.minimum(tl.maximum(L + K - idx * 8, 0), 8)
        clear = tl.where(hi > lo, ((1 << hi) - 1) - ((1 << lo) - 1), 0)
        tl.store(mask_ptr + prev + idx, (255 - clear).to(tl.uint8), mask=idx < nbytes)
'''


def patch_fast_attn(src: str) -> str:
    if MARKER in src:
        return src
    anchor = "\nclass _RoleState:"
    if anchor not in src:
        raise RuntimeError("Could not locate _RoleState in selfspec_fast_attn.py")
    src = src.replace(anchor, KERNEL + "\n" + anchor, 1)

    # buffers (draft role only), sized for the worst case of the planner's kv capacity
    old = "        self.state: Dict[str, _RoleState] = {r: _RoleState(r, self.max_bs, self.kv_cap, self.device) for r in ROLES}"
    new = old + """
        # [TRACK2-DRAFT-MASK] static packed-mask buffers of the draft role (graphs read them at replay)
        self.t2_mask = _T2_DRAFT_MASK
        if self.t2_mask:
            _st = self.state[ROLE_DRAFT]
            _st.t2_mask_buf = torch.zeros(self.K * self.kv_cap // 8 + 8 * self.max_bs + 64, dtype=torch.uint8, device=self.device)
            _st.t2_mask_indptr = torch.zeros(self.max_bs + 1, dtype=torch.int32, device=self.device)
            logger.info(f"[track2-draft-mask] ON: one-pass draft attention (custom mask), mask buffer "
                        f"{_st.t2_mask_buf.numel() / 2**20:.1f} MB")"""
    if old not in src:
        raise RuntimeError("Could not locate the role states in SelfSpecFastPlanner.__init__")
    src = src.replace(old, new, 1)

    # write the mask with the kv metadata (eager, capture and every replay go through _meta)
    old = """        if self.check:
            self._check_meta(st, role, bs, n, req_pool_indices, seq_lens, seq_lens_cpu)
        return st"""
    new = """        if role == ROLE_DRAFT and self.t2_mask:
            _t2_draft_mask_kernel[(bs,)](
                seq_lens,
                st.t2_mask_indptr,
                st.t2_mask_buf,
                K=n,
                BS_POW2=triton.next_power_of_2(max(bs, 2)),
                BLOCK=1024,
            )
        if self.check:
            self._check_meta(st, role, bs, n, req_pool_indices, seq_lens, seq_lens_cpu)
        return st"""
    if old not in src:
        raise RuntimeError("Could not locate the tail of _meta")
    src = src.replace(old, new, 1)

    # metadata: no second (causal) pass when the wrapper carries the mask
    old = """        draft = role == ROLE_DRAFT
        return PrefillMetadata([w], False, False, ss_causal=not draft, ss_seed_causal=draft, ss_K=n)"""
    new = """        draft = role == ROLE_DRAFT
        one_pass = draft and getattr(w, "_custom_mask_buf", None) is not None
        return PrefillMetadata([w], False, False, ss_causal=not draft, ss_seed_causal=draft and not one_pass, ss_K=n)"""
    if old not in src:
        raise RuntimeError("Could not locate _metadata")
    src = src.replace(old, new, 1)

    # eager: attach the mask buffers after planning (FlashInfer's eager plan() resets them)
    old = """        if ref is not None:
            self._cpp_plan(st, w, bs)
            self._check("eager", role, bs, w, ref)
        else:
            self._plan_checked(st, w, role, bs, "eager")
        return self._metadata(role, w, n)"""
    new = """        if ref is not None:
            self._cpp_plan(st, w, bs)
            self._check("eager", role, bs, w, ref)
        else:
            self._plan_checked(st, w, role, bs, "eager")
        if role == ROLE_DRAFT and self.t2_mask:
            w._custom_mask_buf = st.t2_mask_buf
            w._mask_indptr_buf = st.t2_mask_indptr[: bs + 1]
        return self._metadata(role, w, n)"""
    if old not in src:
        raise RuntimeError("Could not locate the tail of plan_eager")
    src = src.replace(old, new, 1)

    # graphs: the draft wrappers are built with the static mask buffers
    old = """            paged_kv_indices_buf=st.kv_indices_dev,
            paged_kv_last_page_len_buf=self.ones[:bs],
        )"""
    new = """            paged_kv_indices_buf=st.kv_indices_dev,
            paged_kv_last_page_len_buf=self.ones[:bs],
            custom_mask_buf=st.t2_mask_buf if (role == ROLE_DRAFT and self.t2_mask) else None,
            mask_indptr_buf=st.t2_mask_indptr[: bs + 1] if (role == ROLE_DRAFT and self.t2_mask) else None,
        )"""
    if old not in src:
        raise RuntimeError("Could not locate the graph wrapper construction in capture()")
    src = src.replace(old, new, 1)
    return patch_draft_kernel(src)


DK_MODULE = '''

# [TRACK2-DRAFT-KERNEL] the one-pass draft attention through a prebuilt flashinfer module (tools/build_draft_attn.py)
# whose kCustom mask is load-free -- row 0 (seed) causal, rows >= 1 (masks) every kv: the same pattern as the packed mask
# above, same plan / tiles / numerics, bitwise equal output. "patch" (default): the mask runs only on the causal mask
# tiles (~1.0x the causal kernel's time instead of ~1.9x); "arith": arithmetic mask on every tile; "" / "off": the packed
# custom-mask kernel. Any load failure -> the packed-mask kernel (logged).
_T2_DRAFT_KERNEL = os.environ.get("TRACK2_DRAFT_KERNEL", "patch").strip().lower()


def _t2_draft_module():
    if _T2_DRAFT_KERNEL in ("", "off", "0", "none"):
        return None
    try:
        import tvm_ffi
        import gravity_ocr

        d = os.environ.get("TRACK2_DRAFT_KERNEL_DIR") or os.path.join(os.path.dirname(gravity_ocr.__file__), "_native")
        so = os.path.join(d, f"fi_draft_seed{_T2_DRAFT_KERNEL}.so")
        if not os.path.isfile(so):
            logger.warning(f"[track2-draft-kernel] {so} missing: packed-mask draft attention")
            return None
        m = tvm_ffi.load_module(so)
        logger.info(f"[track2-draft-kernel] ON: draft attention via {so}")
        print(f"[track2-draft-kernel] ON: seed{_T2_DRAFT_KERNEL} draft attention", flush=True)
        return m
    except Exception as e:  # pragma: no cover
        logger.warning(f"[track2-draft-kernel] load failed ({e}): packed-mask draft attention")
        return None


class _T2DraftModule:
    """flashinfer wrapper module stand-in: plan = the prebuilt module's PrefillPlan (the same C++ as the stock module's),
    paged_run appends the variant's one extra argument (sm_scale, as flashinfer's run() computes it)."""

    def __init__(self, mod, w, head_dim):
        self._m, self._w, self._d = mod, w, head_dim
        self.plan = mod.plan

    def paged_run(self, *args):
        sc = self._w._sm_scale
        if sc is None:
            sc = 1.0 / (self._d ** 0.5)
        return self._m.paged_run(*args, float(sc))
'''


def patch_draft_kernel(src: str) -> str:
    """Route the draft role's wrappers (eager + captured) through the load-free seed-mask module (sub-agent audit)."""
    if "[TRACK2-DRAFT-KERNEL]" in src:
        return src
    anchor = "\nclass _RoleState:"
    if anchor not in src:
        raise RuntimeError("Could not locate _RoleState in selfspec_fast_attn.py")
    src = src.replace(anchor, DK_MODULE + "\n" + anchor, 1)

    old = "        self.graph_wrappers: Dict[Tuple[int, str], object] = {}\n"
    new = old + "        self.t2_dk = _t2_draft_module() if self.t2_mask else None  # [TRACK2-DRAFT-KERNEL]\n"
    if src.count(old) != 1:
        raise RuntimeError("Could not locate graph_wrappers in SelfSpecFastPlanner.__init__")
    src = src.replace(old, new, 1)

    old = """            w = self._Wrapper(self.backend.workspace_buffer, "NHD", backend=self.backend.prefill_backend)
            self._share_int_ws(st, w)
"""
    new = """            w = self._Wrapper(self.backend.workspace_buffer, "NHD", backend=self.backend.prefill_backend)
            if role == ROLE_DRAFT and self.t2_dk is not None:  # [TRACK2-DRAFT-KERNEL]
                w._jit_module = _T2DraftModule(self.t2_dk, w, self.head_dim)
            self._share_int_ws(st, w)
"""
    if src.count(old) != 1:
        raise RuntimeError("Could not locate the eager wrapper construction")
    src = src.replace(old, new, 1)

    old = """            mask_indptr_buf=st.t2_mask_indptr[: bs + 1] if (role == ROLE_DRAFT and self.t2_mask) else None,
        )
        self._share_int_ws(st, w)
"""
    new = """            mask_indptr_buf=st.t2_mask_indptr[: bs + 1] if (role == ROLE_DRAFT and self.t2_mask) else None,
        )
        if role == ROLE_DRAFT and self.t2_dk is not None:  # [TRACK2-DRAFT-KERNEL]
            w._jit_module = _T2DraftModule(self.t2_dk, w, self.head_dim)
        self._share_int_ws(st, w)
"""
    if src.count(old) != 1:
        raise RuntimeError("Could not locate the captured wrapper's _share_int_ws")
    src = src.replace(old, new, 1)
    return src


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-sglang-py", required=True)
    args = ap.parse_args()
    out = Path(args.out_sglang_py) / "sglang/srt/speculative/selfspec_fast_attn.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(patch_fast_attn(Path(args.src).read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    py_compile.compile(str(out), doraise=True)
    print(f"Patched & verified: {out}", flush=True)


if __name__ == "__main__":
    main()
