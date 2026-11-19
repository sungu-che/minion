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


KV_NOSPLIT_HEADER = '''

# [TRACK2-KV-NOSPLIT] graph wrappers that can only ever plan one KV chunk per request run without flashinfer's KV split
# (no merge kernel); "0" = the stock graph plan.
_T2_KV_NOSPLIT = os.environ.get("TRACK2_KV_NOSPLIT", "1") == "1"

# [TRACK2-ATTN-NOFILL] wrappers hand run() an uninitialized output instead of flashinfer's torch.zeros (every element
# is overwritten by the attention/merge kernel); "0" = flashinfer's own allocation.
_T2_ATTN_NOFILL = os.environ.get("TRACK2_ATTN_NOFILL", "1") == "1"


def _t2_nofill_wrapper(base):
    """flashinfer's paged prefill wrapper with run() allocating its output uninitialized."""

    class _T2NoFillWrapper(base):
        def run(self, q, paged_kv_cache, *args, **kwargs):
            if (kwargs.get("out") is None and not kwargs.get("return_lse", False) and kwargs.get("kv_block_scales") is None
                    and isinstance(paged_kv_cache, tuple)):
                kwargs["out"] = torch.empty(q.shape[:-1] + (paged_kv_cache[1].shape[-1],),
                                            dtype=getattr(self, "_cached_o_data_type", None) or q.dtype, device=q.device)
            return base.run(self, q, paged_kv_cache, *args, **kwargs)

    return _T2NoFillWrapper
'''


def patch_kv_nosplit(src: str) -> str:
    """Apply bit-identical KV no-split (bs >= 35 on L40S) and uninitialized attention output allocation."""
    if "[TRACK2-KV-NOSPLIT]" in src:
        return src
    anchor = "\n@triton.jit\ndef _ss_kv_meta_kernel("
    if anchor not in src:
        raise RuntimeError("Could not locate _ss_kv_meta_kernel in selfspec_fast_attn.py")
    src = src.replace(anchor, KV_NOSPLIT_HEADER + "\n" + anchor, 1)

    repls = [
        (
            '        self._Wrapper = BatchPrefillWithPagedKVCacheWrapper\n',
            '        # [TRACK2-ATTN-NOFILL]\n'
            '        self.t2_nofill = _T2_ATTN_NOFILL\n'
            '        self._Wrapper = (_t2_nofill_wrapper(BatchPrefillWithPagedKVCacheWrapper) if self.t2_nofill\n'
            '                         else BatchPrefillWithPagedKVCacheWrapper)\n',
        ),
        (
            '        assert backend.page_size == 1, "SELFSPEC_FASTATTN requires --page-size 1"\n',
            '        assert backend.page_size == 1, "SELFSPEC_FASTATTN requires --page-size 1"\n'
            '        # [TRACK2-KV-NOSPLIT] graph wrappers with T * (bs + 1) > split_bound plan without KV split\n'
            '        self.t2_nosplit = _T2_KV_NOSPLIT\n'
            '        self.t2_split_bound = (2 * int(torch.cuda.get_device_properties(self.device).multi_processor_count)) // self.num_kv_heads\n'
            '        self.t2_ns_graphs: Dict[str, list] = {"nosplit": [], "split": []}\n'
            '        self._t2_ns_reported = False\n'
            '        self._t2_chk = None\n'
            '        self.n_one_chunk_checked = 0\n'
            '        logger.info(f"[track2-kv-nosplit] {\'ON\' if self.t2_nosplit else \'OFF\'}: split bound {self.t2_split_bound}")\n',
        ),
        (
            '    def _split_args(self, graph: bool):\n'
            '        # a fixed split size under CUDA graphs changes split_kv / the plan-info layout vs the captured one -> eager only (warned\n'
            '        # once at startup)\n'
            '        fixed = self.fixed_split if (self.fixed_split > 0 and not graph and not self.no_split) else -1\n'
            '        return fixed, self.no_split\n',
            '    def _split_args(self, w):\n'
            '        no_split = self.no_split or getattr(w, "_ss_no_split", False)\n'
            '        fixed = self.fixed_split if (self.fixed_split > 0 and not w.is_cuda_graph_enabled and not no_split) else -1\n'
            '        return fixed, no_split\n'
            '\n'
            '    # ------------------------------------------------------------------ [TRACK2-KV-NOSPLIT]\n'
            '    def _t2_nosplit_ok(self, bs: int, n: int) -> bool:\n'
            '        if not self.t2_nosplit:\n'
            '            return False\n'
            '        group = self.num_qo_heads // self.num_kv_heads\n'
            '        if not ((bs * n - bs + 1) * group > 64 and self.head_dim < 256):\n'
            '            return False\n'
            '        tiles = -(-(n * group) // 128)\n'
            '        return tiles * (bs + 1) > self.t2_split_bound\n'
            '\n'
            '    def _t2_tiles(self, info, bs: int, n: int) -> int:\n'
            '        return bs * (-(-(n * (self.num_qo_heads // self.num_kv_heads)) // int(info[3])))\n'
            '\n'
            '    def _t2_layout_ok(self, info, bs: int, n: int) -> bool:\n'
            '        info = [int(x) for x in info]\n'
            '        return info[13] == 1 and info[14] == 0 and 0 < self._t2_tiles(info, bs, n) <= info[0]\n'
            '\n'
            '    def _t2_run_info(self, raw, bs: int, n: int):\n'
            '        import tvm_ffi\n'
            '        info = [int(x) for x in raw]\n'
            '        if not self._t2_layout_ok(info, bs, n):\n'
            '            raise RuntimeError(f"[track2-kv-nosplit] bs={bs}: unexpected no-split graph plan info {info}")\n'
            '        info[0] = self._t2_tiles(info, bs, n)\n'
            '        return tvm_ffi.convert(info)\n'
            '\n'
            '    def _t2_check_one_chunk(self, st: _RoleState, w, role: str, bs: int):\n'
            '        if self._t2_chk is None:\n'
            '            ws = torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=self.device)\n'
            '            self._t2_chk = (ws, torch.empty(ws.shape, dtype=torch.uint8, device="cpu", pin_memory=True))\n'
            '        int_ws, pinned = self._t2_chk\n'
            '        n = st.n\n'
            '        torch.cuda.synchronize()\n'
            '        info = [int(x) for x in w._cached_module.plan(\n'
            '            w._float_workspace_buffer, int_ws, pinned, st.qo(n)[0][: bs + 1], st.kv_indptr_host[: bs + 1], st.kvl_host[:bs],\n'
            '            w._max_total_num_rows or bs * n, bs, self.num_qo_heads, self.num_kv_heads, 1, True, self.head_dim, self.head_dim,\n'
            '            False, -1, -1, False, 0)]\n'
            '        host = pinned.numpy()\n'
            '        rows = bs * n\n'
            '        mi = np.frombuffer(host[info[7] : info[7] + 4 * (rows + 1)].tobytes(), dtype=np.int32)\n'
            '        chunk = int(np.frombuffer(host[info[9] : info[9] + 4].tobytes(), dtype=np.int32)[0])\n'
            '        if info[14] != 1 or not bool((np.diff(mi) == 1).all()) or chunk < int(st.kvl_np[:bs].max()):\n'
            '            raise RuntimeError(f"[track2-kv-nosplit] {role} bs={bs}: stock split plan has >1 KV chunk/req")\n'
            '        self.n_one_chunk_checked += 1\n',
        ),
        (
            '        fixed, no_split = self._split_args(graph)\n'
            '        n = st.n\n'
            '        self._wait_pinned(st)\n'
            '        w._plan_info = w._cached_module.plan(\n',
            '        fixed, no_split = self._split_args(w)\n'
            '        n = st.n\n'
            '        self._wait_pinned(st)\n'
            '        raw = w._cached_module.plan(\n',
        ),
        (
            '        w._qo_indptr_last = bs * n\n'
            '        w._batch_size = bs\n',
            '        w._qo_indptr_last = bs * n\n'
            '        w._batch_size = bs\n'
            '        if graph and no_split:\n'
            '            run = getattr(w, "_ss_run_info", None)\n'
            '            if run is None:\n'
            '                run = w._ss_run_info = self._t2_run_info(raw, bs, n)\n'
            '            w._plan_info = run\n'
            '        else:\n'
            '            w._plan_info = raw\n'
            '        return raw\n',
        ),
        (
            '        fixed, no_split = self._split_args(w.is_cuda_graph_enabled)\n',
            '        fixed, no_split = self._split_args(w)\n',
        ),
        (
            '    def _check(self, where, role, bs, w, ref):\n'
            '        got = list(w._plan_info)\n',
            '    def _check(self, where, role, bs, raw, ref):\n'
            '        got = list(raw)\n',
        ),
        (
            '        self._cpp_plan(st, w, bs)\n'
            '        if ref is not None:\n'
            '            self._check(where, role, bs, w, ref)\n'
            '            self.n_checked += 1\n'
            '            if self.n_checked in (1, 10, 100, 1000) or self.n_checked % 10000 == 0:\n'
            '                logger.info(f"[selfspec-check] {self.n_checked} fast-path plans checked (metadata + plan info), 0 mismatches")\n',
            '        raw = self._cpp_plan(st, w, bs)\n'
            '        if ref is not None:\n'
            '            self._check(where, role, bs, raw, ref)\n'
            '            if w.is_cuda_graph_enabled and getattr(w, "_ss_no_split", False):\n'
            '                self._t2_check_one_chunk(st, w, role, bs)\n'
            '            self.n_checked += 1\n'
            '            if self.n_checked in (1, 10, 100, 1000) or self.n_checked % 10000 == 0:\n'
            '                logger.info(f"[selfspec-check] {self.n_checked} fast-path plans checked (metadata + plan info), 0 mismatches")\n'
            '        return raw\n',
        ),
        (
            '        if ref is not None:\n'
            '            self._cpp_plan(st, w, bs)\n'
            '            self._check("eager", role, bs, w, ref)\n',
            '        if ref is not None:\n'
            '            self._check("eager", role, bs, self._cpp_plan(st, w, bs), ref)\n',
        ),
        (
            '        ref = self._python_plan(st, w, bs)\n'
            '        self._cpp_plan(st, w, bs)\n'
            '        self._check("capture", role, bs, w, ref)\n'
            '        self.graph_wrappers[(bs, role)] = w\n'
            '        self.capture_info[(bs, role)] = tuple(int(x) for x in w._plan_info)\n',
            '        w._ss_no_split = self._t2_nosplit_ok(bs, n)\n'
            '        ref = self._python_plan(st, w, bs)\n'
            '        if w._ss_no_split and not self._t2_layout_ok(ref, bs, n):\n'
            '            logger.warning(f"[track2-kv-nosplit] {role} bs={bs}: unexpected no-split plan {ref}: stock split plan")\n'
            '            w._ss_no_split = False\n'
            '            ref = self._python_plan(st, w, bs)\n'
            '        raw = self._cpp_plan(st, w, bs)\n'
            '        self._check("capture", role, bs, raw, ref)\n'
            '        self.graph_wrappers[(bs, role)] = w\n'
            '        self.capture_info[(bs, role)] = tuple(int(x) for x in raw)\n'
            '        no_split = bool(self._split_args(w)[1])\n'
            '        self.t2_ns_graphs["nosplit" if no_split else "split"].append((bs, role))\n'
            '        if no_split:\n'
            '            self._once(("t2-nosplit", role), f"[track2-kv-nosplit] {role} bs={bs}: graph without KV split, "\n'
            '                                             f"{int(w._plan_info[0])} CTAs/KV-head (raw {int(raw[0])})")\n',
        ),
        (
            '        self._plan_checked(st, w, role, bs, "replay")\n'
            '        self._check_capture_layout(w, role, bs)\n',
            '        raw = self._plan_checked(st, w, role, bs, "replay")\n'
            '        self._check_capture_layout(raw, role, bs)\n'
            '        if not self._t2_ns_reported:\n'
            '            self._t2_ns_reported = True\n'
            '            ns, sp = (sorted({b for b, _ in self.t2_ns_graphs[k]}) for k in ("nosplit", "split"))\n'
            '            print(f"[track2-kv-nosplit] {\'ON\' if self.t2_nosplit else \'OFF\'}: {len(self.t2_ns_graphs[\'nosplit\'])} no-split "\n'
            '                  f"(bs {ns}), {len(self.t2_ns_graphs[\'split\'])} split (bs {sp}); bound {self.t2_split_bound}; "\n'
            '                  f"nofill={self.t2_nofill}", flush=True)\n',
        ),
        (
            '    def _check_capture_layout(self, w, role: str, bs: int):\n'
            '        """the replay plan info must be the capture\'s (the graph\'s kernels read the int workspace at its offsets): 3 layout fields on\n'
            '        every replay, all 15 on the first 64 replays of the wrapper, every 1024th, and always with SELFSPEC_CHECK=1."""\n'
            '        key = (bs, role)\n'
            '        cap = self.capture_info[key]\n'
            '        pi = w._plan_info\n',
            '    def _check_capture_layout(self, pi, role: str, bs: int):\n'
            '        """the replay plan info must be the capture\'s (the graph\'s kernels read the int workspace at its offsets): 3 layout fields on\n'
            '        every replay, all 15 on the first 64 replays of the wrapper, every 1024th, and always with SELFSPEC_CHECK=1."""\n'
            '        key = (bs, role)\n'
            '        cap = self.capture_info[key]\n',
        ),
    ]
    for i, (old, new) in enumerate(repls):
        if src.count(old) != 1:
            raise RuntimeError(f"patch_kv_nosplit hunk {i} expected 1 match, got {src.count(old)}: {old[:80]!r}")
        src = src.replace(old, new, 1)
    return src


def patch_draft_kernel(src: str) -> str:
    """Route the draft role's wrappers (eager + captured) through the load-free seed-mask module (sub-agent audit)."""
    if "[TRACK2-DRAFT-KERNEL]" in src:
        return patch_kv_nosplit(src)
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
    return patch_kv_nosplit(src)


def patch_fast_commit(src: str) -> str:
    """Inject ViT launch hiding right between self.event.record() and self.event.synchronize() in SelfSpecFastCommit."""
    marker = "# [TRACK2-VIT-HIDE-LAUNCH]"
    if marker in src:
        return src
    hdr_anchor = "logger = logging.getLogger(__name__)\n"
    if hdr_anchor not in src:
        raise RuntimeError("Could not locate logger in selfspec_fast_commit.py")
    hdr = hdr_anchor + """
# [TRACK2-VIT-HIDE-LAUNCH] hide ViT kernel launch (~9ms/batch) behind decode's event.synchronize() GPU wait
try:
    from gravity_ocr.vit_prefetch import pump_behind_decode as _t2_vit_pump
except Exception:
    _t2_vit_pump = None
"""
    src = src.replace(hdr_anchor, hdr, 1)
    old = "        self.event.record()\n        self.event.synchronize()            # the round's only readback"
    new = (
        "        self.event.record()\n"
        "        if _t2_vit_pump is not None:        # [TRACK2-VIT-HIDE-LAUNCH]\n"
        "            _t2_vit_pump(self.event)\n"
        "        self.event.synchronize()            # the round's only readback"
    )
    cnt = src.count(old)
    if cnt != 2:
        raise RuntimeError(f"Expected 2 self.event.synchronize() sites in selfspec_fast_commit.py, found {cnt}")
    return src.replace(old, new)


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
