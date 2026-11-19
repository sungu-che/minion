"""Whole sheets: blocks, title-block protection, the boilerplate bank and the archive index.

    python tools/page_eval.py --device rocm [--read]

1. Every sheet of the calibration and held-out sets (masters, scans, red-lined
   scans, sheets with other notes, gauge cards) is cut into blocks; each block gets
   the ground-truth kinds it covers (``guide`` = the printed notes).
2. Block vectors (unigram v0, 2-gram v1, 3-gram v2; ``drawing/bank.py``) come from
   one batched vision pass per sheet.
3. The bank is built from the calibration sheets (notes = prejudice, everything
   with content = bias); its threshold is the highest score a calibration content
   block gets when the keys of its own sheet are left out.
4. On the held-out sheets: notes dropped, content blocks dropped (should be 0),
   title / signature blocks protected in both layouts, separability (AUC) of the
   score per level, with and without the double centring.
5. Pattern search: the red-line set (new value, note, stamp) of the calibration
   red-lined sheets, averaged into one query, against every held-out block (AUC,
   R-precision, how many reach the plan's s ≥ 0.85), per level.
6. Archive: each scan is matched to its master among all masters by inner product.
7. With ``--read`` the blocks of the held-out scans and red-lined scans are read
   and every ground-truth text is scored against the blocks over it: the substring
   CER (a block may hold several items) and how many items no block covers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gravity_ocr import score  # noqa: E402
from gravity_ocr.drawing import bank, pipeline  # noqa: E402
from gravity_ocr.model import GravityOCR  # noqa: E402

CONTENT = {"value", "label", "connector", "title", "table_title", "table", "note", "sign", "sign_text", "memo", "new_value", "stamp",
           "form_title", "form", "memo_title"}  # fmt: skip


def overlap(a, b) -> float:
    """Share of box ``b`` covered by box ``a``."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    return ix * iy / max((b[2] - b[0]) * (b[3] - b[1]), 1)


def gt_box(g) -> tuple:
    x0, y0, x1, y1 = g["box"]
    return (int(x0), int(y0), int(x1), int(y1))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    s = np.concatenate([pos, neg])
    ranks = s.argsort().argsort().astype(np.float64) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def protection(sheets: list[dict]) -> dict:
    """Title / signature text in both layouts.  ``safe``: no block that holds a part of the GT
    box (≥ 10 % of the box, or ≥ 50 % of the block inside it) can be dropped; ``unseen``: no block
    there at (the text was not cut out as a block).  Also how many printed-note blocks the rules
    protect (those the bank can no longer drop)."""
    prot = Counter()
    for sh in sheets:
        layout = sh["page"].get("layout", "card")
        for g in sh["page"]["gt"]:
            gb = gt_box(g) if g.get("box") else None
            if gb is None:
                continue
            if g["kind"] in ("title", "sign", "sign_text"):
                over = [b for b in sh["blocks"] if overlap(gb, b.box) >= 0.5 or overlap(b.box, gb) >= 0.1]
                key = f"{layout}:{g['kind']}"
                prot[f"{key}:total"] += 1
                prot[f"{key}:unseen"] += not over
                prot[f"{key}:safe"] += bool(over) and all(b.protected for b in over)
            elif g["kind"] == "guide":
                inside = [b for b in sh["blocks"] if overlap(gb, b.box) >= 0.5]
                prot["guide:blocks"] += len(inside)
                prot["guide:protected_blocks"] += sum(b.protected for b in inside)
    return dict(sorted(prot.items()))


def load_sheets(model, set_name: str) -> list[dict]:
    folder = ROOT / "build" / "synth" / set_name
    pages = json.loads((folder / "pages.json").read_text(encoding="utf-8"))
    out = []
    for p in pages:
        rgb = np.asarray(Image.open(folder / p["image"]).convert("RGB"))
        t0 = time.perf_counter()
        blocks, meas = pipeline.page_blocks(rgb)
        t_blocks = time.perf_counter() - t0
        if model is None:  # blocks and protection only
            out.append({"set": set_name, "page": p, "blocks": blocks, "t_blocks": t_blocks})
            continue
        pipeline.prepare(rgb, blocks)
        t1 = time.perf_counter()
        vecs = bank.sheet_vectors(model, rgb, blocks)
        t_vec = time.perf_counter() - t1
        labels = []
        for b in blocks:
            kinds = {g["kind"] for g in p["gt"] if g.get("box") and overlap(b.box, gt_box(g)) >= 0.5 or (g.get("box") and overlap(gt_box(g), b.box) >= 0.5)}
            labels.append(kinds)
        out.append({"set": set_name, "page": p, "rgb": rgb, "blocks": blocks, "vecs": vecs, "labels": labels,
                    "t_blocks": t_blocks, "t_vectors": t_vec})  # fmt: skip
        print(f"{set_name} {p['id']:<22} {len(blocks):3d} blocks ({t_blocks * 1e3:.0f} ms), vectors {t_vec:.2f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--package", default="", help="a .gocr package instead of the original weights (its head is sliced already)")
    ap.add_argument("--calib", default="calib,calib-dc")
    ap.add_argument("--heldout", default="heldout,heldout-dc")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--blocks-only", action="store_true", help="no model: block cut and title-block protection only")
    ap.add_argument("--save-vectors", default="", help="write every sheet's block vectors, boxes and labels here (.npz)")
    ap.add_argument("--out", default=str(ROOT / "build" / "eval" / "pages.json"))
    args = ap.parse_args()
    if args.blocks_only:
        sheets = sum((load_sheets(None, s) for s in args.calib.split(",") + args.heldout.split(",")), [])
        report = {"protection": protection(sheets), "blocks_ms": round(1e3 * float(np.mean([s["t_blocks"] for s in sheets])), 1)}
        print("protection", report["protection"], f"blocks {report['blocks_ms']} ms/sheet", flush=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    if args.package:
        model = GravityOCR.from_package(args.package, device=args.device)
    else:
        model = GravityOCR.from_pretrained(ROOT / "models", device=args.device)
        ids = ROOT / "build" / "quant" / "head_ids.npy"
        if ids.exists():
            model.slice_head(np.load(ids))
    calib = sum((load_sheets(model, s) for s in args.calib.split(",")), [])
    held = sum((load_sheets(model, s) for s in args.heldout.split(",")), [])
    report: dict = {"device": model.device}
    if args.save_vectors:
        dump = {}
        for i, sh in enumerate(calib + held):
            for level in ("v0", "v1", "v2"):
                dump[f"{i}/{level}"] = sh["vecs"][level]
            dump[f"{i}/boxes"] = np.array([b.box for b in sh["blocks"]], np.int32).reshape(-1, 4)
            dump[f"{i}/protected"] = np.array([b.protected for b in sh["blocks"]], bool)
        meta = [{"set": sh["set"], "id": sh["page"]["id"], "kind": sh["page"]["kind"], "master": sh["page"].get("master"),
                 "drawing": sh["page"].get("drawing"), "layout": sh["page"].get("layout"), "calib": i < len(calib),
                 "labels": [sorted(lbl) for lbl in sh["labels"]], "size": list(sh["rgb"].shape[:2])}
                for i, sh in enumerate(calib + held)]  # fmt: skip
        dump["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
        np.savez_compressed(args.save_vectors, **dump)

    def is_note(lbl):
        return "guide" in lbl and not (lbl & CONTENT)

    def is_content(lbl):
        return bool(lbl & CONTENT)

    for level in ("v0", "v1", "v2"):
        for centred in (True, False):
            keys, pre, sheet_of = [], [], []
            for si, sh in enumerate(calib):
                for v, lbl in zip(sh["vecs"][level], sh["labels"]):
                    if is_note(lbl) or is_content(lbl):
                        keys.append(v)
                        pre.append(is_note(lbl))
                        sheet_of.append(si)
            keys, pre, sheet_of = np.stack(keys), np.array(pre), np.array(sheet_of)
            bk = bank.Bank.build(keys, pre, level)
            if not centred:
                bk = bank.Bank(bank.l2(keys), np.zeros(keys.shape[1], np.float32), pre, np.inf, level)
            # threshold: the highest leave-one-sheet-out score of a calibration content block
            loso = []
            for si in range(len(calib)):
                rows = np.flatnonzero((sheet_of == si) & ~pre)
                if len(rows):
                    loso.append(bk.score(keys[rows] if centred else keys[rows], exclude=(sheet_of == si)[None, :]).max())
            bk.threshold = float(max(loso)) + 1e-4
            notes, contents, dropped_notes, dropped_content, protected_dropped = [], [], 0, 0, 0
            for sh in held:
                s = bk.score(sh["vecs"][level])
                for b, lbl, sc in zip(sh["blocks"], sh["labels"], s):
                    drop = (not b.protected) and sc > bk.threshold
                    if is_note(lbl):
                        notes.append(sc)
                        dropped_notes += drop
                    elif is_content(lbl):
                        contents.append(sc)
                        dropped_content += drop
                    protected_dropped += drop and b.protected
            key = f"{level}{'' if centred else '-plain'}"
            report[key] = {
                "threshold": round(bk.threshold, 4), "auc": round(auc(np.array(notes), np.array(contents)), 4),
                "notes": len(notes), "notes_dropped": int(dropped_notes), "content": len(contents),
                "content_dropped": int(dropped_content), "protected_dropped": int(protected_dropped),
            }  # fmt: skip
            print(key, report[key], flush=True)

    # pattern search (plan 3.2): the red-line set – struck value + new value, the note, the stamp – as
    # one query vector (the mean of those blocks on the calibration red-lined sheets), scored against
    # every block of every held-out sheet by inner product (both centred; tools/archive_eval.py
    # compares the uncentred query and best-match variants)
    red = {"new_value", "note", "stamp"}
    for level in ("v0", "v1", "v2"):
        q = [v for sh in calib for v, lbl in zip(sh["vecs"][level], sh["labels"]) if lbl & red]
        if not q:
            continue
        # centred on the mean calibration block: what every block shares does not count
        c = np.concatenate([sh["vecs"][level] for sh in calib]).mean(axis=0)
        q = bank.l2(np.mean(q, axis=0) - c)
        pos, neg = [], []
        for sh in held:
            for sc, lbl in zip(bank.l2(sh["vecs"][level] - c) @ q, sh["labels"]):
                (pos if lbl & red else neg).append(float(sc))
        pos, neg = np.array(pos), np.array(neg)
        ranked = np.sort(np.concatenate([pos, neg]))[::-1]
        cut = ranked[len(pos) - 1] if len(pos) else np.inf
        report[f"redline_{level}"] = {
            "auc": round(auc(pos, neg), 4), "positives": len(pos), "others": len(neg),
            "r_precision": round(float((pos >= cut).sum() / max(len(pos), 1)), 3),
            "pos_ge_0.85": int((pos >= 0.85).sum()), "others_ge_0.85": int((neg >= 0.85).sum()),
            "pos_mean": round(float(pos.mean()), 3) if len(pos) else None, "others_max": round(float(neg.max()), 3) if len(neg) else None,
        }  # fmt: skip
        print(f"redline_{level}", report[f"redline_{level}"], flush=True)

    # title / signature protection in both layouts
    report["protection"] = protection(held + calib)
    print("protection", report["protection"], flush=True)

    # archive: scan → master, the plan's key (mean of 2-grams) and the block-to-block match on unigrams
    masters = [sh for sh in calib + held if sh["page"]["kind"] == "master"]
    names = [sh["page"]["id"] for sh in masters]
    mat = np.stack([bank.sheet_vector(sh["vecs"]["v2"]) for sh in masters])
    gallery = [sh["vecs"]["v0"] for sh in masters]
    hits = Counter()
    for sh in calib + held:
        p = sh["page"]
        if p.get("master") or p["kind"] == "scan_alt_notice":
            want = p.get("master") or f"{p['drawing']}-master"
            plan = names[int(np.argmax(mat @ bank.sheet_vector(sh["vecs"]["v2"])))]
            match = names[int(np.argmax(bank.archive_scores(sh["vecs"]["v0"], gallery)))]
            hits[f"{p['kind']}:total"] += 1
            hits[f"{p['kind']}:hit_plan_mean_v2"] += plan == want
            hits[f"{p['kind']}:hit_block_match_v0"] += match == want
    report["archive"] = dict(sorted(hits.items()))
    print("archive", report["archive"], flush=True)
    report["timing_ms"] = {
        "blocks_per_sheet": round(1e3 * np.mean([sh["t_blocks"] for sh in calib + held]), 1),
        "vectors_per_sheet": round(1e3 * np.mean([sh["t_vectors"] for sh in calib + held]), 1),
    }  # fmt: skip

    if args.read:
        # every ground-truth text against the blocks over it (covering 30 % of it, or lying mostly
        # inside it).  A block can hold several items (a table, a note beside a label), so the score
        # is the substring CER: the fewest edits from the item to some part of the blocks' text.
        sub, concat, unread = defaultdict(list), defaultdict(list), Counter()
        t0 = time.perf_counter()
        from gravity_ocr.decode import read_many

        texts = []
        for sh in held:
            if sh["page"]["kind"] not in ("scan", "redline"):  # a scan and its red-lined copy (new value, note, stamp)
                continue
            todo = [b for b in sh["blocks"]]
            outs = read_many(model, [(b.prepared.image, b.prepared.prompt) for b in todo], max_tokens=768)
            for b, o in zip(todo, outs):
                b.text = o.text
                texts.append({"sheet": sh["page"]["id"], "box": list(map(int, b.box)), "protected": b.protected, "text": o.text})
            for g in sh["page"]["gt"]:
                if not g.get("text") or g["kind"] not in CONTENT | {"guide"}:
                    continue
                gb = gt_box(g)
                over = [b for b in sh["blocks"] if (overlap(b.box, gb) >= 0.3 or overlap(gb, b.box) >= 0.5) and b.text]
                truth = score.plain(g["text"])
                if not truth:
                    continue
                text = " ".join(score.plain(b.text) for b in sorted(over, key=lambda b: (b.box[1], b.box[0])))
                unread[g["kind"]] += not over
                sub[g["kind"]].append(score.substring_distance(text, truth) / len(truth))
                concat[g["kind"]].append(score.cer(text, truth))
        report["read"] = {k: {"n": len(v), "cer_substring": round(float(np.mean(v)), 4), "cer_whole_blocks": round(float(np.mean(concat[k])), 4),
                              "no_block": unread[k]} for k, v in sorted(sub.items())}  # fmt: skip
        allsub = [x for v in sub.values() for x in v]
        report["read_all"] = {"n": len(allsub), "cer_substring": round(float(np.mean(allsub)), 4), "no_block": sum(unread.values())}
        report["read_blocks"] = texts
        report["read_seconds"] = round(time.perf_counter() - t0, 1)
        print("read", report["read"], report["read_all"], report["read_seconds"], flush=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
