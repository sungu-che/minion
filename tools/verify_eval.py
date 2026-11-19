"""Score the ledger verification on decoded crops.

    python tools/verify_eval.py --report build/eval/heldout-rocm-final.json --data build/synth/heldout

Reads an evaluation report written by ``tools/eval.py`` (texts, tokens and the
decoder's logged alternatives), runs :func:`gravity_ocr.verify.verify` with the
set's ledger, and counts what changed: a fix is *good* when the corrected
span now equals the ground truth there, *bad* (a hallucinated correction)
otherwise.  Candidates the verifier refused for lack of evidence are split the
same way (refused-but-right = a missed correction).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gravity_ocr import score, verify  # noqa: E402
from gravity_ocr.decode import Pick  # noqa: E402
from gravity_ocr.tokenizer import Vocab  # noqa: E402


def picks_of(row: dict) -> list[Pick]:
    alts = {i: tuple((t, p) for t, p in a) for i, a in row.get("alts", [])}
    return [Pick(t, c, alts.get(i, ())) for i, (t, c) in enumerate(zip(row["tokens"], row["conf"]))]


def comparable(text: str) -> str:
    return score.plain(score.normalise(text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", default="models/tokenizer.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    vocab = Vocab.from_file(args.vocab)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    data = Path(args.data)
    samples = {s["id"]: s for s in json.loads((data / "samples.json").read_text(encoding="utf-8"))}
    ledger = json.loads((data / "ledger.json").read_text(encoding="utf-8"))
    codes = [e["kks"] for e in ledger["equipment"]] + [s["tag"] for s in ledger["sensors"]]
    names = [p["name"] for p in ledger["people"]]
    counts = Counter()
    edits_before = edits_after = chars = 0
    details = []
    for row in report["rows"]:
        gt = samples[row["id"]]["gt_text"]
        has_alts = "alts" in row
        rep = verify.verify(row["text"], picks_of(row) if has_alts else None, vocab.piece, codes, names, name_fields=True)
        gt_c = comparable(gt)
        before = score.edit_distance(comparable(row["text"]), gt_c)
        after = score.edit_distance(comparable(rep.text), gt_c)
        edits_before += before
        edits_after += after
        chars += len(gt_c)
        for f in rep.fixes:
            good = f.new in gt and f.raw not in gt
            counts[f"fix_{f.kind}_{f.target}_{'good' if good else 'bad'}"] += 1
            details.append({"id": row["id"], "raw": f.raw, "new": f.new, "kind": f.kind, "target": f.target, "good": good})
        for r in rep.rejected:
            right = r["candidate"] in gt and r["raw"] not in gt
            counts[f"refused_{r['target']}_{'right' if right else 'wrong'}"] += 1
            details.append({"id": row["id"], "refused": r, "candidate_right": right})
        counts["flagged"] += len(rep.flagged)
        counts["rows"] += 1
        counts["rows_with_alts"] += has_alts
    fixes = sum(v for k, v in counts.items() if k.startswith("fix_"))
    bad = sum(v for k, v in counts.items() if k.startswith("fix_") and k.endswith("_bad"))
    summary = {
        "rows": counts["rows"], "fixes": fixes, "bad_fixes": bad,
        "hallucination_rate": round(bad / fixes, 4) if fixes else 0.0,
        "cer_before": round(edits_before / max(chars, 1), 5), "cer_after": round(edits_after / max(chars, 1), 5),
        "counts": dict(sorted(counts.items())),
    }  # fmt: skip
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for d in details:
        print(json.dumps(d, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
