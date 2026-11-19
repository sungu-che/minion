#!/usr/bin/env python3
"""Build the self-spec DRAFT attention modules (flashinfer 0.6.7.post2 JIT, sm_89) into gravity_ocr/_native:

  fi_draft_seedpatch.so  paged prefill kernel + SeedCausal variant, with a 2-line prefill.cuh patch: the kCustom logits
                         mask runs only on the causal mask tiles (the only tiles where this mask masks anything)
  fi_draft_seedarith.so  stock kernel + SeedCausal variant (load-free arithmetic mask on every tile; no header patch)

The variant's mask: row 0 (the seed) is causal (kv_idx < kv_len - qo_len + 1), rows >= 1 (the masks) see every kv --
the same attention pattern as the packed custom mask of tools/patch_sglang_selfspec_mask.py, with the same plan, tiles
and numerics, so the draft attention output is bitwise unchanged (checked: 42/42 shape cases, all 33 rows; seed row ==
the verify's causal row 0). Draft attention ~1.9x -> ~1.0x the causal kernel's time (-150 ms GPU per job on the dev set).

Run inside track2-base (needs nvcc, no GPU):
  /opt/sglang/.venv-sglang/bin/python tools/build_draft_attn.py [--out gravity_ocr/_native]
Derived from the sub-agent audit scripts (dattn.py, dattn_patch.py).
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

D = 128

SEED_DECL = r"""
#include <flashinfer/attention/variant_helper.cuh>
#include <flashinfer/math.cuh>
namespace flashinfer {
// self-spec draft window [seed | mask x (K-1)]: row 0 (seed) is causal (bottom-right aligned: kv_idx <= kv_len - qo_len),
// rows >= 1 (masks) see every kv. Same closure / numerics as DefaultAttention<false, false, false, false>.
struct SeedCausalAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  uint32_t qo_len, kv_len;
  uint32_t window_left;
  uint32_t seed_kv_end;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ SeedCausalAttention(const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    sm_scale_log2 = params.sm_scale * math::log2e;
    window_left = (params.window_left >= 0) ? params.window_left : kv_len;
    seed_kv_end = kv_len + 1 - qo_len;
  }

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return (qo_idx != 0) | (kv_idx < seed_kv_end);
  })
};
}  // namespace flashinfer
"""
ARITH_URI = "t2_draft_seedarith_bf16_h128_v1"
PATCH_URI = "t2_draft_seedpatch_bf16_h128_v1"

OLD_ITER = """    const uint32_t mask_iteration =
        (MASK_MODE == MaskMode::kCausal || MASK_MODE == MaskMode::kMultiItemScoring
             ? min(chunk_size,"""
NEW_ITER = """    const uint32_t mask_iteration =
        (MASK_MODE == MaskMode::kCausal || MASK_MODE == MaskMode::kMultiItemScoring ||
                 MASK_MODE == MaskMode::kCustom /* T2 seedpatch */
             ? min(chunk_size,"""
OLD_LOOP = """      if (MASK_MODE == MaskMode::kCustom) {
        logits_mask<KTraits>(params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
                             kv_idx_base, qo_len, kv_len, chunk_end, group_size, s_frag, tid,
                             kv_head_idx);
      } else {"""
NEW_LOOP = """      if (MASK_MODE == MaskMode::kCustom) {
        if (iter >= mask_iteration) {  // T2 seedpatch: only the causal mask tiles
          logits_mask<KTraits>(params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
                               kv_idx_base, qo_len, kv_len, chunk_end, group_size, s_frag, tid,
                               kv_head_idx);
        }
      } else {"""


def jit_args(uri, name, decl):
    import torch

    return [uri, torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32, D, D, [], [], ["sm_scale"], ["double"], name,
            decl]


def patched_include_tree(root: Path) -> Path:
    import flashinfer.jit.env as je

    dst = root / "include" / "flashinfer"
    shutil.copytree(je.FLASHINFER_INCLUDE_DIR / "flashinfer", dst)
    p = dst / "attention" / "prefill.cuh"
    s = p.read_text()
    assert s.count(OLD_ITER) == 1 and s.count(OLD_LOOP) == 1, "prefill.cuh changed: re-derive the seedpatch hunks"
    p.write_text(s.replace(OLD_ITER, NEW_ITER).replace(OLD_LOOP, NEW_LOOP))
    return root / "include"


def _lib_path(spec, work: Path) -> Path:
    for attr in ("get_library_path", "library_path", "jit_library_path"):
        v = getattr(spec, attr, None)
        v = v() if callable(v) else v
        if v and Path(v).is_file():
            return Path(v)
    hits = sorted(work.rglob(f"{spec.name}.so"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"no built library for {spec.name} under {work}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "gravity_ocr" / "_native"))
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="t2_draft_attn_"))
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(work))
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "8.9")
    os.environ.setdefault("MAX_JOBS", str(os.cpu_count() or 8))
    import flashinfer

    assert flashinfer.__version__ == "0.6.7.post2", f"built against flashinfer 0.6.7.post2, found {flashinfer.__version__}"
    from flashinfer.jit.attention.modules import gen_customize_batch_prefill_module
    from flashinfer.jit.core import gen_jit_spec

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    arith = gen_customize_batch_prefill_module("fa2", *jit_args(ARITH_URI, "SeedCausalAttention", SEED_DECL))
    arith.build(verbose=False)
    shutil.copy2(_lib_path(arith, work), out / "fi_draft_seedarith.so")
    print(f"[draft-attn] seedarith built in {time.time() - t:.0f}s -> {out / 'fi_draft_seedarith.so'}", flush=True)

    t = time.time()
    inc = patched_include_tree(work / "patch")
    decl = SEED_DECL.replace("SeedCausalAttention", "SeedPatchAttention")
    base = gen_customize_batch_prefill_module("fa2", *jit_args(PATCH_URI, "SeedPatchAttention", decl))
    spec = gen_jit_spec(PATCH_URI + "_p", base.sources, extra_include_paths=[str(inc)])
    spec.build(verbose=False)
    shutil.copy2(_lib_path(spec, work), out / "fi_draft_seedpatch.so")
    print(f"[draft-attn] seedpatch built in {time.time() - t:.0f}s -> {out / 'fi_draft_seedpatch.so'}", flush=True)


if __name__ == "__main__":
    main()
