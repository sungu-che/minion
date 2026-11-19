#!/usr/bin/env python3
"""Patch stock SGLang `glm_ocr.py` (extracted from track2-base:v1) with Track2AllInVramAccelerator.

Usage (on the GPU box after extracting `/opt/sglang/python/sglang/srt/models/glm_ocr.py`):
  python3 tools/patch_sglang_glm_ocr.py \
      --src ~/base-src/sglang/python/sglang/srt/models/glm_ocr.py \
      --out ~/my-team/gravityocr/sglang/python/sglang/srt/models/glm_ocr.py
"""
from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path

MARKER = "# [TRACK2-MINION-ACCEL]"

INJECT_IMPORT = f"""{MARKER}
import os as _t2_os
try:
    from gravityocr.track2_accel import Track2AllInVramAccelerator as _Track2Accel
except Exception:
    _Track2Accel = None
"""


def patch_source(src_text: str) -> str:
    if MARKER in src_text:
        return src_text
    lines = src_text.splitlines(keepends=True)
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_idx = i
            break
    lines.insert(insert_idx, INJECT_IMPORT + "\n")
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare patched SGLang glm_ocr.py overlay for Track 2")
    ap.add_argument("--src", required=True, help="Path to stock /opt/sglang/python/sglang/srt/models/glm_ocr.py")
    ap.add_argument("--out", required=True, help="Destination path in gravityocr/sglang/python/sglang/srt/models/glm_ocr.py")
    args = ap.parse_args()

    src_path = Path(args.src)
    out_path = Path(args.out)
    if not src_path.is_file():
        sys.exit(f"Source glm_ocr.py not found: {src_path}")

    patched = patch_source(src_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(patched, encoding="utf-8", newline="\n")
    py_compile.compile(str(out_path), doraise=True)
    print(f"Patched and syntax-verified: {out_path}", flush=True)


if __name__ == "__main__":
    main()
