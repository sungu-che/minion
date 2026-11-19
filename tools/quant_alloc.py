"""Turn the screening results into a precision per group.

    python tools/quant_alloc.py --budget 0.002 --out build/quant/alloc.json

Every group starts at q8 (near-lossless; a group not screened at q8 is taken as
ΔNLL 0 there).  Moving a group to a cheaper scheme saves bytes and costs the
ΔNLL the screen measured for that group alone; the moves are taken greedily by
bytes saved per unit of ΔNLL (a ΔNLL below ``--floor`` counts as ``--floor``:
within the noise every move costs a little), as long as the summed ΔNLL stays
within ``--budget`` and the move's flip rate stays within ``--max-flips``.  A
scheme that was not screened for a group is never chosen for it.  The sizes
come from the quantised packages themselves (``build/quant/<scheme>.gocr``), so
they do not depend on what was screened.  The sum assumes the groups add up; the
result is a hypothesis that ``quant_sweep.py config`` (teacher forcing on every
calibration crop) and a full decode then check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from gravity_ocr.container import Package  # noqa: E402
from quant_sweep import QUANT, groups  # noqa: E402

LADDER = ["q8", "q6", "q5", "q4", "nf4", "q3"]  # the first is the starting point; any smaller one is a move


def sizes(gs: dict[str, list[str]], ladder: list[str]) -> dict[str, dict[str, int]]:
    """Stored bytes of every group under every scheme of the ladder."""
    out: dict[str, dict[str, int]] = {g: {} for g in gs}
    for s in ladder:
        path = QUANT / f"{s}.gocr"
        if not path.exists():
            continue
        pkg = Package(path)
        for g, names in gs.items():
            out[g][s] = sum(pkg.stored_bytes(n) for n in names)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default=str(QUANT / "screen.json"))
    ap.add_argument("--budget", type=float, default=0.002, help="summed ΔNLL (nats per token) allowed")
    ap.add_argument("--floor", type=float, default=2e-5)
    ap.add_argument("--max-flips", type=float, default=0.004, help="a move whose flip rate is above this is not taken")
    ap.add_argument("--keep", default="", help="comma list of groups that stay at q8")
    ap.add_argument("--ladder", default=",".join(LADDER), help="schemes considered; the first is where every group starts")
    ap.add_argument("--out", default=str(QUANT / "alloc.json"))
    args = ap.parse_args()
    ladder = args.ladder.split(",")
    res = json.loads(Path(args.screen).read_text(encoding="utf-8"))
    gs = groups()
    size = sizes(gs, ladder)
    screened: dict[str, dict[str, dict]] = {g: {} for g in gs}
    for key, r in res.items():
        if "@" in key:
            g, s = key.split("@")
            if g in screened:
                screened[g][s] = r
    keep = set(filter(None, args.keep.split(",")))
    top = ladder[0]
    assign = {g: top for g in gs}
    cost = {g: max(screened[g].get(top, {}).get("dnll", 0.0), 0.0) for g in gs}
    spent = sum(cost.values())
    while True:
        best = None
        for g in gs:
            if g in keep:
                continue
            cur = assign[g]
            for s in ladder:
                r = screened[g].get(s)
                if r is None or s not in size[g] or r["flips"] > args.max_flips:
                    continue
                d = max(r["dnll"], args.floor) - cost[g]
                saved = size[g][cur] - size[g][s]
                if saved <= 0 or spent + max(d, 0) > args.budget:
                    continue
                ratio = saved / max(d, args.floor)
                if best is None or ratio > best[0]:
                    best = (ratio, g, s, d)
        if best is None:
            break
        _, g, s, d = best
        assign[g] = s
        cost[g] += d
        spent += max(d, 0)
    per = {s: sum(1 for v in assign.values() if v == s) for s in ladder}
    nbytes = sum(size[g][s] for g, s in assign.items())
    full = sum(size[g][top] for g in gs)
    print(f"summed ΔNLL {spent:.5f} of {args.budget}; groups per scheme {per}; "
          f"matrices {nbytes / 2**20:.1f} MB (all {top}: {full / 2**20:.1f} MB)")  # fmt: skip
    for g in gs:
        r = screened[g].get(assign[g], {})
        print(f"  {g:<18} {assign[g]:<3} {size[g][assign[g]] / 2**20:7.1f} MB  dnll {r.get('dnll', 0.0):+.5f}")
    Path(args.out).write_text(json.dumps(assign, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
