"""Archive retrieval and pattern search on saved sheet vectors (``page_eval.py --save-vectors``).

    python tools/archive_eval.py build/eval/pages-vectors.npz

Archive (plan Part 6): every scan, red-lined scan and scan with another notice is matched
against all masters; top-1 accuracy per method and level.

``mean``      the plan: the normalised mean of the sheet's block vectors, inner product;
``centred``   the same after removing the mean sheet vector of the gallery (what all sheets share);
``chamfer``   block to block: for every block of the query its best match among the master's
              blocks, averaged (both ways), on block vectors centred on the gallery's mean block.

Pattern search: the red-line set of the calibration sheets as one query (mean of its blocks)
or as its blocks (best match), against every held-out block; AUC and R-precision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RED = {"new_value", "note", "stamp"}


def l2(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    s = np.concatenate([pos, neg])
    ranks = s.argsort().argsort().astype(np.float64) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load(path: str) -> tuple[list[dict], dict]:
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    for i, m in enumerate(meta):
        for level in ("v0", "v1", "v2"):
            m[level] = z[f"{i}/{level}"]
        m["boxes"] = z[f"{i}/boxes"]
    return meta, z


def archive(sheets: list[dict], level: str, method: str) -> tuple[int, int]:
    masters = [s for s in sheets if s["kind"] == "master"]
    queries = [s for s in sheets if s["kind"] in ("scan", "redline", "scan_alt_notice")]
    if method in ("mean", "centred"):
        mv = np.stack([l2(s[level].mean(axis=0)) for s in masters])
        qv = np.stack([l2(s[level].mean(axis=0)) for s in queries])
        if method == "centred":
            c = mv.mean(axis=0)
            mv, qv = l2(mv - c), l2(qv - c)
        sims = qv @ mv.T
    else:  # chamfer on centred block vectors
        c = np.concatenate([s[level] for s in masters]).mean(axis=0)
        mb = [l2(s[level] - c) for s in masters]
        sims = np.zeros((len(queries), len(masters)))
        for i, q in enumerate(queries):
            qb = l2(q[level] - c)
            for j, m in enumerate(mb):
                s = qb @ m.T
                sims[i, j] = 0.5 * (s.max(axis=1).mean() + s.max(axis=0).mean())
    hit = 0
    for i, q in enumerate(queries):
        want = q["master"] or f"{q['drawing']}-master"
        hit += masters[int(np.argmax(sims[i]))]["id"] == want
    return hit, len(queries)


def pattern(sheets: list[dict], level: str, method: str) -> dict:
    cal = [s for s in sheets if s["calib"]]
    held = [s for s in sheets if not s["calib"]]
    keys = np.concatenate([s[level][[bool(set(lbl) & RED) for lbl in s["labels"]]] for s in cal])
    allv = np.concatenate([s[level] for s in held])
    lab = np.concatenate([[bool(set(lbl) & RED) for lbl in s["labels"]] for s in held])
    c = np.concatenate([s[level] for s in cal]).mean(axis=0) if method.endswith("centred") else 0.0
    if method.startswith("mean"):
        score = l2(allv - c) @ l2(keys.mean(axis=0) - c)
    else:  # best match against the calibration red-line blocks
        score = (l2(allv - c) @ l2(keys - c).T).max(axis=1)
    pos, neg = score[lab], score[~lab]
    cut = np.sort(score)[::-1][max(int(lab.sum()) - 1, 0)]
    return {"auc": round(auc(pos, neg), 4), "r_precision": round(float((pos >= cut).sum() / max(len(pos), 1)), 3), "positives": int(lab.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vectors")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sheets, _ = load(args.vectors)
    report = {"archive": {}, "pattern": {}}
    for level in ("v0", "v1", "v2"):
        for method in ("mean", "centred", "chamfer"):
            h, n = archive(sheets, level, method)
            report["archive"][f"{method}/{level}"] = f"{h}/{n}"
        for method in ("mean", "mean-centred", "best", "best-centred"):
            report["pattern"][f"{method}/{level}"] = pattern(sheets, level, method)
    for k, v in report["archive"].items():
        print(f"archive {k:<14} {v}")
    for k, v in report["pattern"].items():
        print(f"pattern {k:<18} {v}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
