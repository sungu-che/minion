"""The output slice (``gravity_ocr/surgery.py``): what it keeps, what the sets need, what English loses.

    python tools/slice_eval.py [--write build/quant/head_ids_v4.npy]

1. Tier sizes and kept / total per token class.
2. Coverage of every set's text ground truth (crops and sheets; symbol class names excluded).
3. English: half of the engineering lexicon (``surgery.ENGLISH``) goes into the domain corpus, the
   other half is held out; a word form (CAPS, Title, lower, each with and without a leading space)
   is lost when its tokenisation needs a row the slice cut.  The slice in ``build/quant/head_ids.npy``
   (the one the quantisation sweep used) is measured the same way for comparison.
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

SETS = ("calib", "heldout", "calib-dc", "heldout-dc")


def texts_of(name: str) -> list[str]:
    folder = ROOT / "build" / "synth" / name
    out = [s["gt_text"] for s in json.loads((folder / "samples.json").read_text(encoding="utf-8"))]
    for p in json.loads((folder / "pages.json").read_text(encoding="utf-8")):
        out += [g["text"] for g in p["gt"] if g.get("text") and g["kind"] != "symbol"]  # symbol GT = class names
    return out


def lost_forms(vocab: Vocab, keep: set, words: list[str]) -> dict:
    out = {"CAPS": 0, "Title": 0, "lower": 0, "forms": 0}
    for w in words:
        for kind, form in zip(("CAPS", "Title", "lower") * 2, surgery.english_forms([w])):
            out["forms"] += 1
            out[kind] += any(t not in keep for t in vocab.encode(form))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default="", help="save the slice built with the whole lexicon here")
    args = ap.parse_args()
    vocab = Vocab.from_file(ROOT / "models" / "tokenizer.json")
    calib = [s["gt_text"] for n in ("calib", "calib-dc") for s in json.loads((ROOT / "build/synth" / n / "samples.json").read_text(encoding="utf-8"))]
    words = list(dict.fromkeys(surgery.ENGLISH.split()))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(words))
    fit = [words[i] for i in order[: len(words) // 2]]
    held = [words[i] for i in order[len(words) // 2 :]]
    sl_fit = surgery.build(vocab, surgery.domain_corpus(calib, english=fit))
    full = surgery.build(vocab, surgery.domain_corpus(calib))
    old_path = ROOT / "build" / "quant" / "head_ids.npy"
    rows = [("this slice, half the lexicon", set(sl_fit.ids.tolist()))]
    if old_path.exists():
        rows.insert(0, ("build/quant/head_ids.npy", set(np.load(old_path).tolist())))
    print(f"tiers (whole lexicon): {full.tiers}")
    for c, (k, t) in full.classes.items():
        print(f"  {c:<8} kept {k:6d} of {t:6d}")
    sets = {n: texts_of(n) for n in SETS if (ROOT / "build/synth" / n / "pages.json").exists()}
    print(f"\nlexicon {len(words)} words: {len(fit)} in the corpus, {len(held)} held out")
    for label, keep in rows + [("this slice, whole lexicon", set(full.ids.tolist()))]:
        miss = {n: sum(1 for t in tx for tok in vocab.encode(t) if tok not in keep) for n, tx in sets.items()}
        print(f"{label:<30} held-out {lost_forms(vocab, keep, held)}  fit {lost_forms(vocab, keep, fit)}  missing set tokens {miss}")
    if args.write:
        np.save(args.write, full.ids)
        print(f"wrote {args.write} ({full.size} ids)")


if __name__ == "__main__":
    main()
