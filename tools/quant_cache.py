"""Quantise every weight matrix once per scheme: the material of the sensitivity sweep.

    python tools/quant_cache.py --schemes q8,q6,q5,q4,q3,nf4 --workers 5

Writes ``build/quant/<scheme>.gocr`` holding all quantisable matrices of the
model (vision blocks, downsample, merger, language layers, the sliced output
head) in that one scheme, plus ``build/quant/head_ids.npy``.  The sweep then
swaps groups of tensors between the original weights and these packages.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gravity_ocr import language, quant, surgery, vision  # noqa: E402
from gravity_ocr.container import PackageWriter, SafetensorsFile  # noqa: E402
from gravity_ocr.tokenizer import Vocab  # noqa: E402

OUT = ROOT / "build" / "quant"


def matrices() -> list[str]:
    """Names of the tensors whose precision the sweep decides."""
    names = []
    for n in vision.tensor_names():
        short = n[len(vision.PREFIX) :]
        if vision.is_matrix(short) and short != "patch_embed.proj.weight":
            names.append(n)
    for n in language.tensor_names():
        if language.is_matrix(n[len(language.PREFIX) :]):
            names.append(n)
    return names + [language.HEAD]


def head_ids() -> np.ndarray:
    path = OUT / "head_ids.npy"
    if path.exists():
        return np.load(path)
    vocab = Vocab.from_file(ROOT / "models" / "tokenizer.json")
    texts = []
    for name in ("calib", "calib-dc"):
        p = ROOT / "build" / "synth" / name / "samples.json"
        if p.exists():
            texts += [s["gt_text"] for s in json.loads(p.read_text(encoding="utf-8"))]
    ids = surgery.build(vocab, surgery.domain_corpus(texts)).ids
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(path, ids)
    return ids


def _job(args):
    name, scheme = args
    src = SafetensorsFile(ROOT / "models" / "model.safetensors")
    if name == language.HEAD:
        w = src.rows(name, np.load(OUT / "head_ids.npy"))
    else:
        w = src.get(name)
    t0 = time.perf_counter()
    parts = quant.quantize(w, scheme)
    return name, scheme, parts, tuple(w.shape), time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="q8,q6,q5,q4,q3,nf4")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    head_ids()
    names = matrices()
    for scheme in args.schemes.split(","):
        path = OUT / f"{scheme}.gocr"
        if path.exists():
            print(f"{path} exists, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        tmp = path.with_suffix(".part")
        with PackageWriter(tmp, {"scheme": scheme, "source": "models/model.safetensors"}) as wr:
            with ProcessPoolExecutor(args.workers) as pool:
                for i, (name, sch, parts, shape, sec) in enumerate(pool.map(_job, [(n, scheme) for n in names])):
                    wr.add_parts(name, parts, sch, shape)
                    if i % 40 == 0:
                        print(f"  {scheme} {i + 1}/{len(names)} {name} {sec:.1f}s", flush=True)
        tmp.replace(path)
        print(f"{scheme}: {len(names)} tensors in {time.perf_counter() - t0:.0f}s -> {path} ({path.stat().st_size / 2**20:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
