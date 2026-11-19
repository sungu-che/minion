#!/usr/bin/env python3
"""Build Track 2 Byte-BPE Safe task core vocabulary slices (table 16K, text 16K, formula 8K).

Usage:
  python3 tools/build_vocab_slices.py --models /opt/gravityocr/weights --out vocab_slices
  python3 tools/build_vocab_slices.py --models /opt/gravityocr/weights --out build/vocab_slices --with-rest-gguf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gravity_ocr import surgery  # noqa: E402
from gravity_ocr.tokenizer import Vocab  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Track 2 core_table / core_text / core_formula vocab slices")
    ap.add_argument("--models", required=True, help="Model directory containing tokenizer.json (and model.safetensors)")
    ap.add_argument("--out", default=str(ROOT / "vocab_slices"), help="Output directory for *.npy slices")
    ap.add_argument("--table-budget", type=int, default=16_384)
    ap.add_argument("--text-budget", type=int, default=16_384)
    ap.add_argument("--formula-budget", type=int, default=8_192)
    ap.add_argument("--with-rest-gguf", action="store_true", help="Also build Q4_K lm_head_rest.gguf if GPU lib is built")
    args = ap.parse_args()

    models = Path(args.models)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok_path = models / "tokenizer.json"
    if not tok_path.is_file():
        sys.exit(f"tokenizer.json not found in {models}")

    vocab = Vocab.from_file(tok_path)
    slices = surgery.build_task_slices(
        vocab,
        table_budget=args.table_budget,
        text_budget=args.text_budget,
        formula_budget=args.formula_budget,
    )

    meta: dict[str, dict] = {"vocab_size": {"total": vocab.size}}
    for task, sl in slices.items():
        path = out / f"core_{task}_ids.npy"
        np.save(path, sl.ids.astype(np.int64))
        meta[task] = {"size": sl.size, "tiers": sl.tiers, "file": path.name}
        print(f"  wrote {path} ({sl.size} ids, tiers={sl.tiers})", flush=True)

    # Optionally build Q4_K lm_head_rest.gguf for the union-complement cut rows (for Candle GPU rescue)
    if args.with_rest_gguf and (models / "model.safetensors").is_file():
        try:
            from gravity_ocr import gguf, language
            from gravity_ocr.container import SafetensorsFile

            union_ids = np.unique(np.concatenate([sl.ids for sl in slices.values()]))
            rest_ids = np.setdiff1d(np.arange(vocab.size, dtype=np.int64), union_ids)
            np.save(out / "rest_ids.npy", rest_ids)

            src = SafetensorsFile(models / "model.safetensors")
            full_head = np.asarray(src.get(language.HEAD), np.float32)
            rest_w = np.ascontiguousarray(full_head[rest_ids], np.float32)
            blocks = gguf.quantize("q4_k", {"w": rest_w}, "f32", rest_w.shape)
            gguf_path = out / "lm_head_rest.gguf"
            entry = gguf.Entry("lm_head.rest.weight", rest_w.shape, "q4_k", lambda: blocks)
            gguf.write(gguf_path, {"general.architecture": "gravity-ocr", "gravity.rest_rows": int(rest_ids.size)}, [entry])
            meta["rest_gguf"] = {"rows": int(rest_ids.size), "file": gguf_path.name, "bytes": gguf_path.stat().st_size}
            print(f"  wrote {gguf_path} ({rest_ids.size} rows, {gguf_path.stat().st_size / (1 << 20):.1f} MB)", flush=True)
        except Exception as e:
            print(f"  [warn] skipped lm_head_rest.gguf ({e}); PyTorch VRAM rescue remains active", flush=True)

    (out / "slices_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Done: {out}", flush=True)


if __name__ == "__main__":
    main()
