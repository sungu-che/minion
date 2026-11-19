"""Write the final model package (``.gocr``).

    python tools/build_model.py --assign build/quant/final.json --out build/model/gravity-ocr-ko.gocr

``--assign`` maps tensor names (or sweep group names, see ``tools/quant_sweep.py``)
to a scheme; matrices it does not name keep ``--default``.  The quantised parts
are copied from ``build/quant/<scheme>.gocr`` (``tools/quant_cache.py``) when they
are there, so the package holds exactly what the sweep measured.  Norms, biases
and the patch embedding are stored as float32; the output head is the 16,384-row
slice (``lm_head.ids`` maps rows to token ids; ``--head-ids`` picks the slice) and
the rows it cut are stored apart (``lm_head.rest``, ``--rest`` scheme): the runtime
checks every pass that none of them beats the kept rows.  The input embedding is
stored in any scheme (``--embed``; rows are gathered superblock by superblock).  The
tokenizer, the model configuration, the block-diffusion settings, the measured
quantisation plan and (optionally) the UnifiedNgram table travel in the same file.
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

from gravity_ocr import language, quant, vision  # noqa: E402
from gravity_ocr.container import Package, PackageWriter, SafetensorsFile  # noqa: E402
from gravity_ocr.model import HEAD_IDS, REST, REST_IDS  # noqa: E402

QUANT = ROOT / "build" / "quant"


def expand(assign: dict, default: str) -> dict[str, str]:
    sys.path.insert(0, str(ROOT / "tools"))
    from quant_sweep import groups  # noqa: E402

    gs = groups()
    out = {n: default for n in sum(gs.values(), [])}
    for key, scheme in assign.items():
        for n in gs.get(key, [key]):
            out[n] = scheme
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--assign", default="", help="JSON {tensor or group: scheme}")
    ap.add_argument("--default", default="q8")
    ap.add_argument("--embed", default="q8", choices=("q8", "q6", "q5", "q4", "f16", "f32"),
                    help="scheme of the input embedding (rows are gathered per token; quant_sweep.py embed measures it)")  # fmt: skip
    ap.add_argument("--head-ids", default=str(QUANT / "head_ids.npy"), help="the output slice (full token ids, ascending)")
    ap.add_argument("--rest", default="q4", help="scheme of the output rows the slice cut (checked every pass), or 'none'")
    ap.add_argument("--ngram", default=str(ROOT / "build" / "ngram" / "table.gocr"))
    ap.add_argument("--no-ngram", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "build" / "model" / "gravity-ocr-ko.gocr"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    folder = Path(args.models)
    src = SafetensorsFile(folder / "model.safetensors")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    bd_path = folder / "block_diffusion.json"
    bd = json.loads(bd_path.read_text(encoding="utf-8")) if bd_path.exists() else None
    assign = expand(json.loads(Path(args.assign).read_text(encoding="utf-8")) if args.assign else {}, args.default)
    head_ids = np.load(args.head_ids)
    # the quantisation caches hold the head sliced by build/quant/head_ids.npy; another slice is quantised here
    cached_head = (QUANT / "head_ids.npy").exists() and np.array_equal(np.load(QUANT / "head_ids.npy"), head_ids)
    caches: dict[str, Package] = {}

    def cached(scheme: str) -> Package | None:
        if scheme not in caches:
            p = QUANT / f"{scheme}.gocr"
            caches[scheme] = Package(p) if p.exists() else None
        return caches[scheme]

    plan = {"default": args.default, "embed": args.embed, "head_rows": int(head_ids.size), "tensors": {}}
    meta = {"config": config, "block_diffusion": bd, "quant": plan, "format": "GOCRPK01", "source": "zai-org GLM-OCR weights"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    total = 0
    with PackageWriter(tmp, meta) as w:
        w.add_blob("tokenizer.json", (folder / "tokenizer.json").read_bytes())
        names = vision.tensor_names() + language.tensor_names()
        for name in names + [language.HEAD]:
            scheme = assign.get(name)
            if scheme is None:  # norms, biases, patch embedding
                arr = src.get(name)
                total += w.add(name, arr, "f32")
                plan["tensors"][name] = "f32"
                continue
            pkg = cached(scheme)
            if pkg is not None and name in pkg and (name != language.HEAD or cached_head):
                total += w.add_parts(name, pkg.parts(name), pkg.scheme(name), pkg.shape(name))
            else:
                arr = src.rows(name, head_ids) if name == language.HEAD else src.get(name)
                total += w.add(name, arr, scheme)
            plan["tensors"][name] = scheme
        total += w.add_array(HEAD_IDS, head_ids.astype(np.int64))
        if args.rest != "none":
            rest = np.setdiff1d(np.arange(src.shape(language.HEAD)[0], dtype=np.int64), head_ids)
            total += w.add(REST, src.rows(language.HEAD, rest), args.rest)
            total += w.add_array(REST_IDS, rest)
            plan["tensors"][REST] = args.rest
        total += w.add(language.EMBED, src.get(language.EMBED), args.embed)
        plan["tensors"][language.EMBED] = args.embed
        if not args.no_ngram and Path(args.ngram).exists():
            ng = Package(args.ngram)
            for n in ng.names():
                total += w.add_parts(n, ng.parts(n), ng.scheme(n), ng.shape(n))
            w.meta["ngram"] = {"prefix": "ngram.", "rows": int((np.asarray(ng.get("ngram.tags")) != 0).sum())}
    tmp.replace(out)
    size = out.stat().st_size
    print(f"wrote {out} ({size / 2**20:.1f} MB, tensors {total / 2**20:.1f} MB) in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
