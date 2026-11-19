#!/usr/bin/env python3
"""Multi-layer run report + cross-layer checks from gravity_ocr/t2log.py output.

  python3 tools/t2log_report.py <team_process_*.log | run log> [more logs ...] [--dir <TRACK2_LOG_DIR>]

Per job (one manifest the client sent): the L1..L4 summary lines are matched on the shared perf_counter clock, then
  * checks  : crops == L2 images == L3 prefilled requests == L3 finished == L4 encoded images; no empty / failed
              outputs; no on-demand ViT encodes; socket retries reported
  * timeline: when the tokenizer manager delivered the last image, first / last scheduler batch, last ViT launch,
              client wall time, all relative to the client's first send
  * cost    : prefill / decode / gap seconds, tokens per request-round, ViT GPU ms
  * per task: completion quantiles; the last finishers
With --dir (TRACK2_LOG >= 2) also: slowest requests per task and decode batch-size buckets.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

LINE = re.compile(r"\[t2log (L\d)\] (\{.*\})\s*$")


def parse(paths):
    out = defaultdict(list)
    for p in paths:
        for line in open(p, encoding="utf-8", errors="replace"):
            m = LINE.search(line)
            if m:
                try:
                    out[m.group(1)].append(json.loads(m.group(2)))
                except json.JSONDecodeError:
                    pass
    return out


def window_of(rec, layer):
    if layer == "L2":
        return rec.get("first_in"), rec.get("last_out")
    if layer == "L3":
        return rec.get("t_first"), rec.get("t_last")
    if layer == "L4":
        return rec.get("first_launch"), rec.get("last_launch")
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--dir", default=None, help="TRACK2_LOG_DIR with the level-2 JSONL files")
    a = ap.parse_args()
    S = parse(a.logs)
    jobs = S.get("L1", [])
    if not jobs:
        raise SystemExit("no [t2log L1] lines found (TRACK2_LOG=0?)")
    ok_all = True
    jobs = sorted(jobs, key=lambda x: x["t_sent0"])
    for k, j in enumerate(jobs):
        t0 = j["t_sent0"]
        t_end = j["t_start"] + j["wall_s"] + 2.0
        if k + 1 < len(jobs):   # back-to-back jobs: a layer summary belongs to the job it starts in
            t_end = min(t_end, jobs[k + 1]["t_sent0"] - 0.05)
        # timed jobs ("<n>-test") are checked strictly; the server's warm-up pass(es) on the dev set ("dev") may run the
        # manifest more than once inside one client job, so their counts are reported as information only
        strict = str(j["job"]).endswith("-test")
        match = {}
        for layer in ("L2", "L3", "L4"):
            cands = []
            for rec in S.get(layer, []):
                lo, hi = window_of(rec, layer)
                # a layer's job starts after the client's first send (the IPC probe / server warmup before it are separate);
                # the tokenizer's burst may also hold the probe image sent just before: L2 matches on overlap
                if lo is None:
                    continue
                if (layer == "L2" and hi is not None and hi >= t0 and lo <= t_end) or (t0 - 0.05 <= lo <= t_end):
                    cands.append(rec)
            match[layer] = cands
        L2, L3, L4 = match["L2"], match["L3"], match["L4"]
        n2 = sum(r["images"] for r in L2)
        n3p = sum(r["prefill"]["reqs"] for r in L3)
        n3f = sum(r["finished"] for r in L3)
        n4 = sum(r["images"] for r in L4)
        od = sum(r["ondemand_images"] for r in L4)
        print(f"\n=== job {j['job']}: {j['crops']} crops, client wall {j['wall_s']:.3f}s (prep {j['prep_s']:.3f}s)"
              + ("" if strict else "  [warm-up: layer counts are information only]"))
        checks = [
            ("L1 no empty outputs", j["empty"] == 0, f"empty={j['empty']}"),
            ("L1 no failed requests", j["failed"] == 0, f"failed={j['failed']} retries={j['retries']} "
                                                         f"socket_failures={j['socket_failures']}"),
            ("L2 tokenizer saw every crop", (not L2) or n2 >= j["crops"], f"L2 images={n2}" if L2 else "L2: no summary"),
            # a request split by chunked prefill shows up in two prefill batches: >= is the invariant
            ("L3 prefilled every crop", (not L3) or n3p >= j["crops"], f"L3 prefill reqs={n3p}" if L3 else "L3: no summary"),
            ("L3 finished every crop", (not L3) or n3f == j["crops"], f"L3 finished={n3f}" if L3 else ""),
            ("L4 encoded every crop", (not L4) or n4 == j["crops"], f"L4 images={n4}" if L4 else "L4: no summary"),
            ("L4 no on-demand ViT", od == 0, f"on-demand images={od}"),
        ]
        for name, ok, info in checks:
            if strict or name.startswith("L1"):
                ok_all &= bool(ok)
                tag = "OK " if ok else "BAD"
            else:
                tag = "-- "
            print(f"  [{tag}] {name:30s} {info}")
        print("  timeline (s after the first send):")
        if L2:
            print(f"    L2 tokenizer: first image in {min(r['first_in'] for r in L2) - t0:+.3f}, last image out "
                  f"{max(r['last_out'] for r in L2) - t0:+.3f}  (latency ms p50/p90/max {L2[0]['latency_ms_p50_p90_max']}, "
                  f"downscaled {sum(r['downscaled'] for r in L2)}, patches {sum(r['patches'] for r in L2)})")
        if L4:
            print(f"    L4 ViT     : first launch {min(r['first_launch'] for r in L4) - t0:+.3f}, last launch "
                  f"{max(r['last_launch'] for r in L4) - t0:+.3f}, {sum(r['batches'] for r in L4)} batches, "
                  f"{sum(r['patches'] for r in L4)} patches, GPU {sum(r['gpu_ms_sum'] for r in L4):.0f} ms, host launch "
                  f"{sum(r['host_launch_ms'] for r in L4):.0f} ms, wait_first {sum(r['wait_first'] for r in L4)} "
                  f"({sum(r['wait_first_ms'] for r in L4):.0f} ms)")
        if L3:
            r = L3[0] if len(L3) == 1 else max(L3, key=lambda x: x["iterations"])
            print(f"    L3 scheduler: first batch {min(x['t_first'] for x in L3) - t0:+.3f}, last batch "
                  f"{max(x['t_last'] for x in L3) - t0:+.3f}, iterations {sum(x['iterations'] for x in L3)}")
            pf, dc = r["prefill"], r["decode"]
            print(f"    L3 cost     : prefill n={pf['n']} tok={pf['tokens']} run {pf['run_s']:.3f}s + result {pf['result_s']:.3f}s | "
                  f"decode n={dc['n']} mean_bs={dc['mean_bs']} tok/req-round={dc['tokens_per_request_round']} run "
                  f"{dc['run_s']:.3f}s + result {dc['result_s']:.3f}s | gaps {r['gaps_s']:.3f}s (ms p50/p90/max {r['gap_ms_p50_p90_max']})")
        print(f"    L1 client   : wall {j['wall_s']:.3f}s, send span {j['send_span_s']}s, normalized outputs {j['normalized']}")
        for t, v in j.get("by_task", {}).items():
            print(f"      {t:8s} n={v['n']:4d} done p50/p90/max {v['done_s_p50_p90_max']}")
        print("      last finishers:", j.get("last"))
    if a.dir:
        recs = defaultdict(list)
        for p in glob.glob(os.path.join(a.dir, "L*_*.jsonl")):
            layer = os.path.basename(p).split("_")[0]
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    recs[layer].append(json.loads(line))
        l1 = recs.get("L1", [])
        if l1:
            last_job = l1[-1]["job"]
            rows = [r for r in l1 if r["job"] == last_job]
            t_s = min(r["send"] for r in rows)
            print(f"\n=== level-2 detail for job {last_job}: slowest crops per task (done s after first send)")
            for t in ("table", "text", "formula"):
                rs = sorted((r for r in rows if r["task"] == t), key=lambda r: -r["done"])[:5]
                print(f"  {t}: " + ", ".join(f"{r['id'][:28]}(rank {r['rank']}, area {r['area']}, {r['done'] - t_s:.2f}s)" for r in rs))
        l3 = recs.get("L3", [])
        if l3:
            b = defaultdict(list)
            for r in l3:
                if r["mode"] != "EXTEND":
                    b[min(96, (r["bs"] + 7) // 8 * 8)].append(r["t2"] - r["t0"])
            print("  decode ms/iteration by batch-size bucket:",
                  {k: f"{1e3 * sum(v) / len(v):.1f}ms x{len(v)}" for k, v in sorted(b.items())})
    print(f"\nall checks {'PASSED' if ok_all else 'FAILED'}")


if __name__ == "__main__":
    main()
