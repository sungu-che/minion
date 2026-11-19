"""Bake the UnifiedNgram table and measure what its layer-0 residual does.

    python tools/ngram_eval.py --gates 0,0.04,0.08,0.16

The table is baked on the CPU (native kernels) from the calibration ground truth,
the calibration ledgers' codes and the domain corpus, then the GPU bench of
``tools/quant_sweep.py`` scores the calibration crops by teacher forcing with
the residual switched on at each gate: mean NLL of the reference tokens and the
share of flipped argmaxes, against gate 0 (no residual).  It also counts how
many output positions find their trigram in the table (tag matches) and how many
land in an occupied bucket of another trigram (rejected by the tag).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gravity_ocr import ngram  # noqa: E402
from gravity_ocr.container import PackageWriter  # noqa: E402
from gravity_ocr.model import GravityOCR  # noqa: E402

OUT = ROOT / "build" / "ngram"


def seed_corpus() -> tuple[list[str], list[str]]:
    texts, codes = [], []
    for name in ("calib", "calib-dc"):
        folder = ROOT / "build" / "synth" / name
        texts += [s["gt_text"] for s in json.loads((folder / "samples.json").read_text(encoding="utf-8"))]
        led = json.loads((folder / "ledger.json").read_text(encoding="utf-8"))
        codes += [e["kks"] for e in led["equipment"]] + [s["tag"] for s in led["sensors"]]
    return texts, codes


def bake() -> ngram.NgramTable:
    path = OUT / "table.gocr"
    if path.exists():
        from gravity_ocr.container import Package

        return ngram.NgramTable.load(Package(path))
    texts, codes = seed_corpus()
    model = GravityOCR.from_pretrained(ROOT / "models", device="cpu")
    t0 = time.perf_counter()
    table = ngram.build(model, ngram.seed_texts(codes, texts) + ngram.code_texts(codes))
    OUT.mkdir(parents=True, exist_ok=True)
    with PackageWriter(path, {"kind": "unified_ngram", "rank": ngram.RANK, "buckets": ngram.BUCKETS}) as w:
        table.save(w)
    print(f"baked {table.rows} rows in {time.perf_counter() - t0:.1f}s, {table.nbytes / 2**20:.1f} MB", flush=True)
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="0,0.04,0.08,0.16")
    ap.add_argument("--sets", default="calib,calib-dc")
    args = ap.parse_args()
    table = bake()
    sys.path.insert(0, str(ROOT / "tools"))
    from quant_sweep import Bench  # noqa: E402

    bench = Bench(args.sets.split(","))
    hits = misses = positions = 0
    injects = []
    for s in bench.samples:
        tgt = s["target"]
        rows = np.zeros((len(tgt) - 1, 1536), np.float32) if len(tgt) > 1 else None
        for j in range(2, len(tgt) - 1):
            a, b, c = tgt[j - 2], tgt[j - 1], tgt[j]
            positions += 1
            i = ngram.bucket_of(a, b, c)
            if int(table.tags[i]) == ngram.tag_of(a, b, c):
                hits += 1
                z = table.coef[i].astype(np.float32) * table.scale[i]
                rows[j] = z @ (table.proj + table.learned)
            elif int(table.tags[i]) != 0:
                misses += 1
        injects.append(rows)
    print(f"output positions {positions}: tag matches {hits}, other trigram in the bucket (rejected) {misses}", flush=True)
    results = {"rows": table.rows, "bytes": table.nbytes, "positions": positions, "hits": hits, "rejected": misses}
    for g in (float(x) for x in args.gates.split(",")):
        res = bench.score(bench.ref_tokens, inject=[None if r is None else r * g for r in injects])
        res["dnll"] = res["nll"] - bench.base["nll"]
        results[f"gate={g}"] = res
        print(f"gate {g:5.2f}: nll {res['nll']:.5f} (Δ {res['dnll']:+.5f}) flips {res['flips']:.4f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
