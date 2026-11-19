"""Kernel-level timings: one weight product at a range of row counts, per backend and path.

    python tools/bench_kernels.py                     # numpy, native CPU, every GPU path found
    python tools/bench_kernels.py --out build/bench/kernels.json

Rows of ``x`` stand for: 1 = a decoding step, 8-32 = a draft / verification
pass, hundreds = a prefill, thousands = the vision tower.  The weight shapes are
the real ones of the model (language gate/up, vision qkv, language down, and the
output head whole, sliced to 16,384 rows and the 42,899 rows the slice cut – checked
every pass for a winner – the head only at decoding row counts).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gravity_ocr import gpu, native, quant  # noqa: E402

SHAPES = {"lm.gate_up": (9216, 1536), "vis.qkv": (3072, 1024), "lm.down": (1536, 4608),
          "lm.head": (59283, 1536), "lm.head.sliced": (16384, 1536), "lm.head.rest": (42899, 1536)}  # fmt: skip
ROWS = (1, 4, 16, 32, 64, 128, 512, 2048, 8192)
HEAD_ROWS = (1, 4, 16, 32)  # the output head only sees the rows whose logits are wanted


def timeit(fn, reps: int) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="q4,q6,q8,bf16")
    ap.add_argument("--rows", default=",".join(map(str, ROWS)))
    ap.add_argument("--numpy", action="store_true", help="also time the numpy path (slow)")
    ap.add_argument("--out", default="build/bench/kernels.json")
    args = ap.parse_args()
    rows = [int(r) for r in args.rows.split(",")]
    rng = np.random.default_rng(0)
    k = native.load()
    e = gpu.load("rocm")
    results = []
    for scheme in args.schemes.split(","):
        for wname, shape in SHAPES.items():
            w = (rng.standard_normal(shape) * 0.02).astype(np.float32)
            if scheme == "bf16":
                parts = {"w": (w.view(np.uint32) >> 16).astype(np.uint16)}
            else:
                parts = quant.quantize(w, scheme)
            lin = native.Linear(parts, scheme, shape)
            dense = None
            print(f"== {scheme} {wname} {shape}", flush=True)
            for n in rows:
                if wname.startswith("lm.head") and n not in HEAD_ROWS:
                    continue
                flop = 2.0 * n * shape[0] * shape[1]
                rec = {"scheme": scheme, "weight": wname, "shape": list(shape), "rows": n}
                x = rng.standard_normal((n, shape[1])).astype(np.float32)
                reps = max(1, min(50, int(2e9 / flop)))
                if args.numpy and n <= 2048:
                    if dense is None:
                        dense = quant.dequantize_np(parts, scheme, shape) if scheme != "bf16" else (parts["w"].astype(np.uint32) << 16).view(np.float32)
                    rec["numpy_ms"] = timeit(lambda: x @ dense.T, reps) * 1e3
                if k is not None and n <= 2048:
                    rec["cpu_ms"] = timeit(lambda: k.linear(lin, x), reps) * 1e3
                if e is not None:
                    lib = e.lib.lib
                    for path, name in ((1, "gemv"), (2, "qgemm"), (3, "blas"), (0, "auto")):
                        if path == 1 and n > 32:
                            continue
                        s = lib.gg_bench_linear(e.ptr, lin.addr, n, path, max(3, reps))
                        if s > 0:
                            rec[f"gpu_{name}_ms"] = s * 1e3
                line = "  ".join(f"{key}={val:8.3f}" for key, val in rec.items() if key.endswith("_ms"))
                best = min((v for kk, v in rec.items() if kk.endswith("_ms")), default=None)
                tf = f"{flop / best / 1e9:6.2f} TFLOP/s best" if best else ""
                print(f"n={n:5d} {line}  {tf}", flush=True)
                results.append(rec)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": e.info if e else None, "cpu_threads": k.threads if k else None, "results": results}, indent=1), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
