#!/usr/bin/env python3
"""Rebuild the team's SGLang overlay (gravityocr/sglang/python/...) from the base image's patched SGLang sources.

The base sources are /opt/sglang/python inside track2-base:v1 (SGLang 0.5.10 + gravityocr_selfspec.patch + trida's
new files); a copy works too (e.g. ~/base-full/sglang/python made with the README's docker cp command).

  python3 tools/build_overlay.py --base /opt/sglang/python --out <team>/gravityocr/sglang/python [--no-scheduler]

Every patcher is applied to the pristine base file, in a fixed order, so the overlay is reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import py_compile
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REL = {
    "vision": "sglang/srt/layers/attention/vision.py",
    "runner": "sglang/srt/multimodal/glm_vit_graph_runner.py",
    "glm_ocr": "sglang/srt/models/glm_ocr.py",
    "glm4v_proc": "sglang/srt/multimodal/processors/glm4v.py",
    "logits": "sglang/srt/layers/logits_processor.py",
    "worker": "sglang/srt/speculative/selfspec_diffusion_worker.py",
    "ngram": "sglang/srt/speculative/ngram_info.py",
    "graph": "sglang/srt/model_executor/cuda_graph_runner.py",
    "scheduler": "sglang/srt/managers/scheduler.py",
    "out_proc": "sglang/srt/managers/scheduler_output_processor_mixin.py",
    "fast_attn": "sglang/srt/speculative/selfspec_fast_attn.py",
    "fast_commit": "sglang/srt/speculative/selfspec_fast_commit.py",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="patched SGLang python/ root of the base image (/opt/sglang/python)")
    ap.add_argument("--out", required=True, help="<team>/gravityocr/sglang/python")
    ap.add_argument("--no-scheduler", action="store_true", help="leave scheduler.py to the image (no prefill coalescing)")
    args = ap.parse_args()
    base, out = Path(args.base), Path(args.out)

    vit = _load("patch_sglang_vit_l40s")
    lmh = _load("patch_sglang_lm_head")
    fmm = _load("patch_sglang_fast_mm")
    sch = _load("patch_sglang_scheduler")
    msk = _load("patch_sglang_selfspec_mask")

    def src(key: str) -> str:
        return (base / REL[key]).read_text(encoding="utf-8")

    files: dict[str, str] = {
        "vision": vit.patch_vision_py(src("vision")),
        "runner": vit.patch_runner_py(src("runner")),
        "glm_ocr": vit.patch_glm_ocr_py(src("glm_ocr")),
        "glm4v_proc": fmm.FAST_GLM4V_PY,
        "logits": lmh.patch_logits_processor(src("logits")),
        "worker": lmh.patch_worker(src("worker")),
        "ngram": lmh.patch_ngram_info(src("ngram")),
        "graph": lmh.patch_cuda_graph_runner(src("graph")),
        "fast_attn": msk.patch_fast_attn(src("fast_attn")),
        "fast_commit": msk.patch_fast_commit(src("fast_commit")),
    }
    if not args.no_scheduler:
        files["scheduler"] = sch.patch_scheduler(src("scheduler"))
        files["out_proc"] = sch.patch_output_processor(src("out_proc"))
    for key, text in files.items():
        dst = out / REL[key]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8", newline="\n")
        py_compile.compile(str(dst), doraise=True)
        print(f"{hashlib.md5(text.encode()).hexdigest()[:8]}  {dst}", flush=True)
    stale = out / REL["scheduler"]
    if args.no_scheduler and stale.exists():
        stale.unlink()
        print(f"removed {stale}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
