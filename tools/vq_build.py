"""Build the VQ codebook of simple merge groups from the calibration crops.

    python tools/vq_build.py --device rocm --out build/vq/codebook.gocr

Every calibration crop is prepared as for reading and run through the vision tower;
for its simple groups (blank or line-only, ``vq.simple_groups``) the ``patch_embed``
key and the tower's output token are collected, clustered and summarised
(``vq.build``).  Printed: how many groups are simple, how many words are enabled,
and the hit rate on the calibration groups.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gravity_ocr import preprocess, prune, vq  # noqa: E402
from gravity_ocr.container import PackageWriter  # noqa: E402
from gravity_ocr.drawing import crops  # noqa: E402
from gravity_ocr.model import GravityOCR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--package", default="", help="build from this package's vision tower (the one that will use the codebook)")
    ap.add_argument("--sets", default="calib,calib-dc")
    ap.add_argument("--words", type=int, default=vq.WORDS)
    ap.add_argument("--out", default=str(ROOT / "build" / "vq" / "codebook.gocr"))
    args = ap.parse_args()
    if args.package:
        model = GravityOCR.from_package(args.package, device=args.device)
    else:
        model = GravityOCR.from_pretrained(ROOT / "models", device=args.device)
    keys, values = [], []
    groups = simple_n = blank_n = 0
    t0 = time.perf_counter()
    for name in args.sets.split(","):
        folder = ROOT / "build" / "synth" / name
        for s in json.loads((folder / "samples.json").read_text(encoding="utf-8")):
            prep = crops.prepare_crop(Image.open(folder / s["image"]).convert("RGB"), prompt=s["prompt"], text_px=30)
            rgb = preprocess.fit_image(prep.image)
            patches, gg = preprocess.to_patches(rgb)
            toks = model.vision.forward(patches, gg)
            simple = vq.simple_groups(rgb)
            blank = prune.blank_groups(rgb)
            line = simple & ~blank
            groups += simple.size
            simple_n += int(simple.sum())
            blank_n += int(blank.sum())
            if line.any():
                keys.append(vq.group_keys(model, patches[prune.kept_rows(~line)]))  # the rows of the line groups
                values.append(toks[line])
    keys = np.concatenate(keys)
    values = np.concatenate(values)
    print(f"{groups} groups: blank {blank_n} ({blank_n / groups:.1%}), line-only {len(keys)} ({len(keys) / groups:.1%}); "
          f"collected in {time.perf_counter() - t0:.0f}s", flush=True)  # fmt: skip
    t0 = time.perf_counter()
    book = vq.build(keys, values, args.words)
    word, hit = book.lookup(keys)
    err = np.linalg.norm(book.values[word[hit]] - values[hit], axis=1) / np.maximum(np.linalg.norm(values[hit], axis=1), 1e-6)
    print(f"{book.keys.shape[0]} words, enabled {int(book.enabled.sum())}; calibration line groups hit {hit.mean():.1%}, "
          f"relative token error of the hits {float(err.mean()) if len(err) else 0:.3f}; built in {time.perf_counter() - t0:.0f}s", flush=True)  # fmt: skip
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PackageWriter(out, {"kind": "vq_codebook", "words": int(book.keys.shape[0]), "enabled": int(book.enabled.sum())}) as w:
        book.save(w)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
