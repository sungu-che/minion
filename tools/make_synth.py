"""Generate the synthetic sets: crop scenarios and whole pages, with ground truth.

    python tools/make_synth.py --out build/synth

Two sets are written from two different ledgers.  ``calib`` is what anything
that is fitted may look at (quantisation sensitivity, banks, thresholds);
``heldout`` is only ever scored.  The two share no equipment code.  With
``--aidc`` two more sets come from AI data-centre ledgers (DCIM codes):
``calib-dc`` and ``heldout-dc``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gravity_ocr.synth import crops, drawings, fonts, scan  # noqa: E402
from gravity_ocr.synth.ledger import make_ledger  # noqa: E402


def write_set(out: Path, name: str, seed: int, scale: float, pages: bool, domain: str = "plant") -> dict:
    led = make_ledger(seed, domain=domain)
    folder = out / name
    (folder / "crops").mkdir(parents=True, exist_ok=True)
    (folder / "ledger.json").write_text(json.dumps(led.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
    counts = {k: max(1, round(v * scale)) for k, v in crops.DEFAULT_COUNTS.items()}
    records = []
    for s in crops.make_set(led, seed, counts, prefix=f"{name}-"):
        rel = f"crops/{s.id}.png"
        s.image.save(folder / rel)
        records.append(
            {"id": s.id, "scenario": s.scenario, "image": rel, "prompt": s.prompt, "gt_text": s.gt_text, "gt_cells": s.gt_cells, "meta": s.meta}
        )
    (folder / "samples.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    page_records = []
    if pages:
        (folder / "pages").mkdir(exist_ok=True)
        rng = random.Random(f"{seed}:pages")

        def save(page_id: str, kind: str, img, gt, extra: dict):
            img.save(folder / "pages" / f"{page_id}.png")
            rec = {"id": page_id, "kind": kind, "image": f"pages/{page_id}.png", "size": [img.width, img.height], "gt": gt, **extra}
            page_records.append(rec)

        for d in led.drawings:
            # the master as issued (clean), the same sheet scanned, and a red-lined scan
            c, info = drawings.draw_master(led, d.id, random.Random(f"{seed}:{d.id}"))
            save(f"{d.id}-master", "master", c.img, c.gt, {"drawing": d.id, "layout": d.layout, "info": info})
            img, gt, deg = scan.scanned(c.img, c.gt, rng)
            save(f"{d.id}-scan", "scan", img, gt, {"drawing": d.id, "layout": d.layout, "deg": deg, "master": f"{d.id}-master"})
            c2, info2 = drawings.draw_master(led, d.id, random.Random(f"{seed}:{d.id}"))
            n_before = len(c2.gt)
            marks = drawings.redline(c2, info2, led, d.id, rng)
            if marks:
                for g in c2.gt[n_before:]:
                    g["new"] = True
                img, gt, deg = scan.scanned(c2.img, c2.gt, rng, max_deg=0.5)
                save(f"{d.id}-redline", "redline", img, gt, {"drawing": d.id, "layout": d.layout, "deg": deg, "master": f"{d.id}-master", "marks": marks})
            # the same sheet with different boilerplate, for the filter generalisation test
            c3, _ = drawings.draw_master(led, d.id, random.Random(f"{seed}:{d.id}"), notices=drawings.NOTICES_ALT)
            img, gt, deg = scan.scanned(c3.img, c3.gt, rng)
            save(f"{d.id}-altnotice", "scan_alt_notice", img, gt, {"drawing": d.id, "layout": d.layout, "deg": deg})
        for i in range(3):
            c, info = drawings.gauge_card(led, rng)
            img, gt, deg = scan.scanned(c.img, c.gt, rng)
            save(f"card-{i}", "gauge_card", img, gt, {"deg": deg, "info": info})
        (folder / "pages.json").write_text(json.dumps(page_records, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"seed": seed, "domain": domain, "crops": len(records), "pages": len(page_records), "codes": led.codes, "drawings": [d.id for d in led.drawings]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/synth")
    ap.add_argument("--calib-seed", type=int, default=1101)
    ap.add_argument("--heldout-seed", type=int, default=2207)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplier on the per-scenario sample counts")
    ap.add_argument("--no-pages", action="store_true")
    ap.add_argument("--aidc", action="store_true", help="also write the AI data-centre sets (DCIM codes)")
    ap.add_argument("--only-aidc", action="store_true", help="write only the AI data-centre sets")
    ap.add_argument("--aidc-seeds", default="3301,4409")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"fonts": fonts.used(), "sets": {}}
    jobs = [] if args.only_aidc else [("calib", args.calib_seed, "plant"), ("heldout", args.heldout_seed, "plant")]
    if args.aidc or args.only_aidc:
        a, b = (int(x) for x in args.aidc_seeds.split(","))
        jobs += [("calib-dc", a, "aidc"), ("heldout-dc", b, "aidc")]
    for name, seed, domain in jobs:
        manifest["sets"][name] = write_set(out, name, seed, args.scale, not args.no_pages, domain)
    sets = manifest["sets"]
    for x, y in (("calib", "heldout"), ("calib-dc", "heldout-dc")):
        if x in sets and y in sets:
            manifest.setdefault("shared_codes", {})
            if not isinstance(manifest["shared_codes"], dict):
                manifest["shared_codes"] = {"calib/heldout": manifest["shared_codes"]}
            manifest["shared_codes"][f"{x}/{y}"] = sorted(set(sets[x]["codes"]) & set(sets[y]["codes"]))
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, m in manifest["sets"].items():
        print(f"{name}: seed {m['seed']} ({m.get('domain', 'plant')}), {m['crops']} crops, {m['pages']} pages, drawings {m['drawings']}")
    print("shared equipment codes:", manifest.get("shared_codes"))


if __name__ == "__main__":
    main()
