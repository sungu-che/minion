"""Build the GGUF "pyramid" model: every matrix of the original BF16 checkpoint quantised once to
the GGML block format its measured sensitivity allows, then multiplied as it is on the GPU.

    python tools/build_gguf.py                          # measured plan → build/model/gravity-ocr-ko.gguf
    python tools/build_gguf.py --plan q8                # every matrix Q8_0 (or q6 / q5 / q4)
    python tools/build_gguf.py --plan my.json           # {group or tensor name: scheme}

The measured plan is the allocation of ``docs/QUANT.md`` (53 groups × q8 … q3 screened by
teacher-forced ΔNLL on the GPU, ΔNLL budget 0.002 – the one the ``.gocr`` package uses).  Its
schemes map to GGML as blueprint-sandbox ``ai::pyramid`` does: q8 → Q8_0, q6 → Q6_K,
q5 → Q5_K, q4 / nf4 / q3 → Q4_K (no Q3_K).  Groups are the ones of ``tools/quant_sweep.py``.

Kept as they are: norms and biases (float32), the patch embedding (float32: its rows of 588 are
not whole blocks, and the two frames are summed), the input embedding table (``--embed``,
bf16 by default: it stays on the host, only the rows of a prompt are read).

Quantisation runs on the CPU (``gg_ggml_quantize``: candle's k-quant encoders, row chunks in
parallel); the GPU library must be built (``tools/build_kernels.py``).  The written file is read
back with candle's own GGUF reader as a check.
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

from gravity_ocr import gguf, language, native, pyramid, vision  # noqa: E402
from gravity_ocr.container import SafetensorsFile  # noqa: E402

TO_GGML = {**pyramid.GGML, "f32": "f32", "f16": "f16", "bf16": "bf16", **{k: k for k in gguf.GGML}}
MB = 1 << 20


def plan_of(arg: str) -> tuple[str, dict[str, str]]:
    """``(name, {tensor: ggml kind})`` for the matrices the plan names."""
    gs = pyramid.groups()
    if arg in ("measured", ""):
        name, assign = "measured-0.002", pyramid.MEASURED
    elif arg in ("q8", "q6", "q5", "q4"):
        name, assign = f"all-{arg}", {g: arg for g in gs}
    else:
        name, assign = Path(arg).stem, json.loads(Path(arg).read_text(encoding="utf-8"))
    out = {}
    for key, scheme in assign.items():
        kind = TO_GGML[scheme]
        for t in gs.get(key, [key]):
            out[t] = kind
    return name, out


def source_info(models: Path) -> dict:
    """File, size and (when downloaded with huggingface_hub) the commit of the checkpoint."""
    st = models / "model.safetensors"
    info = {"file": st.name, "bytes": st.stat().st_size}
    meta = models / ".cache" / "huggingface" / "download" / "model.safetensors.metadata"
    if meta.is_file():
        info["commit"] = meta.read_text(encoding="utf-8").splitlines()[0].strip()
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--plan", default="measured", help="measured | q8 | q6 | q5 | q4 | path to {group or tensor: scheme} json")
    ap.add_argument("--default", default="q8_0", help="GGML kind of a matrix the plan does not name")
    ap.add_argument("--embed", default="bf16", choices=("bf16", "f16", "q8_0"), help="input embedding table (host side)")
    ap.add_argument("--out", default=str(ROOT / "build" / "model" / "gravity-ocr-ko.gguf"))
    args = ap.parse_args()

    models = Path(args.models)
    src = SafetensorsFile(models / "model.safetensors")
    plan_name, plan = plan_of(args.plan)
    V, L = vision, language

    entries, kinds = [], {}

    def matrix(name: str, value=None):
        """A matrix entry: its kind from the plan when the rows hold whole blocks."""
        short = name[len(V.PREFIX) :] if name.startswith(V.PREFIX) else None
        reshaped = short is not None and short in V.RESHAPED
        shape = V.reshape_weight(short, np.zeros(src.shape(name), np.float32)).shape if reshaped else src.shape(name)
        kind = plan.get(name, args.default)
        if not gguf.fits(kind, shape):
            kind = "q8_0" if gguf.fits("q8_0", shape) and kind != "f32" and not reshaped else "f32"
        if kind == "bf16" and (reshaped or src.scheme(name) != "bf16"):
            kind = "f32"

        def produce():
            if reshaped:
                arr = np.ascontiguousarray(V.reshape_weight(short, np.asarray(src.get(name), np.float32)), np.float32)
                parts, scheme = {"w": arr}, "f32"
            else:
                parts, scheme, _ = native.parts_of(src, name)
            if kind == "f32":
                out = np.ascontiguousarray(arr if reshaped else src.get(name), np.float32)
            elif kind == "bf16":
                out = np.ascontiguousarray(parts["w"])
            elif kind == "f16":
                out = np.ascontiguousarray(np.asarray(src.get(name), np.float32).astype("<f2"))
            else:
                out = gguf.quantize(kind, parts, scheme, shape)
            src.release_pages(name)
            return out

        kinds[name] = kind
        entries.append(gguf.Entry(name, tuple(shape), kind, produce))

    def vector(name: str):
        kinds[name] = "f32"
        entries.append(gguf.Entry(name, tuple(src.shape(name)), "f32", lambda: np.ascontiguousarray(src.get(name), np.float32)))

    for name in V.tensor_names():
        (matrix if V.is_matrix(name[len(V.PREFIX) :]) else vector)(name)
    for name in L.tensor_names():
        (matrix if L.is_matrix(name[len(L.PREFIX) :]) else vector)(name)
    matrix(L.HEAD)
    emb = args.embed

    def embed():
        if emb == "bf16" and src.scheme(L.EMBED) == "bf16":
            return np.ascontiguousarray(native.parts_of(src, L.EMBED)[0]["w"])
        if emb == "q8_0":
            return gguf.quantize("q8_0", *native.parts_of(src, L.EMBED))
        return np.ascontiguousarray(np.asarray(src.get(L.EMBED), np.float32).astype("<f2" if emb == "f16" else "<f4"))

    kinds[L.EMBED] = emb
    entries.append(gguf.Entry(L.EMBED, tuple(src.shape(L.EMBED)), emb, embed))

    def text(name: str) -> str | None:
        p = models / name
        return p.read_text(encoding="utf-8") if p.is_file() else None

    meta = {
        "general.architecture": "gravity-ocr",
        "general.name": "GravityOCR-KO",
        "general.quantization_version": 2,
        "gravity.reshaped": True,
        "gravity.plan_name": plan_name,
        "gravity.plan": json.dumps(kinds, ensure_ascii=False),
        "gravity.source": json.dumps(source_info(models)),
        "gravity.config": text("config.json"),
        "gravity.tokenizer": text("tokenizer.json"),
    }
    if text("block_diffusion.json") is not None:
        meta["gravity.block_diffusion"] = text("block_diffusion.json")
    meta = {k: v for k, v in meta.items() if v is not None}

    t0 = time.perf_counter()
    done = {"n": 0, "bytes": 0}

    def log(e: gguf.Entry):
        done["n"] += 1
        done["bytes"] += e.nbytes
        if done["n"] % 40 == 0 or done["n"] == len(entries):
            print(f"  {done['n']:4d}/{len(entries)} tensors  {done['bytes'] / MB:8.1f} MB  {time.perf_counter() - t0:6.1f}s", flush=True)

    print(f"plan {plan_name}: {len(entries)} tensors → {args.out}", flush=True)
    total = gguf.write(args.out, meta, entries, log=log)
    by_kind: dict[str, float] = {}
    for e in entries:
        if e.name != L.EMBED:
            by_kind[e.kind] = by_kind.get(e.kind, 0) + e.nbytes / MB
    device_mb = sum(by_kind.values())
    print(f"written {total / MB:.1f} MB in {time.perf_counter() - t0:.1f}s; on the device {device_mb:.1f} MB "
          f"({', '.join(f'{k} {v:.1f}' for k, v in sorted(by_kind.items()))}); embedding {emb} on the host")  # fmt: skip
    n, k = gguf.check_with_candle(args.out)
    print(f"candle's GGUF reader: {n} tensors, {k} metadata entries")


if __name__ == "__main__":
    main()
