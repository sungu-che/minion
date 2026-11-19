"""Benchmark of the backends, one fresh process each, and the report tables.

    python tools/bench.py --package build/model/gravity-ocr-ko.gocr
    python tools/bench.py --backends rocm,cpu --per-scenario 2

Backends (``--backends``, in this order by default):

``hf``      ① the original model in ``transformers`` (torch, CPU, float32 compute on
            the BF16 checkpoint) – the reference implementation, greedy decoding;
``numpy``   ② the Python / numpy runtime on the quantised package;
``cpu``     ③ the same with the Rust CPU kernels (packed weights, SIMD, threads);
``rocm``    ④ the GPU engine on ROCm (HIP kernels on the packed weights);
``vulkan``  ④' the GPU engine on Vulkan (portable path);
``rocm@orig`` the GPU engine on the original BF16 weights (full output head) – what
            the package changes on the same kernels (size, VRAM, speed, accuracy).

Every backend reads the same crops (``--per-scenario`` of each scenario of every
``--sets``), prepared the same way (text zoomed to 30 px, no damping).  Recorded:
load time, vision / prefill / decode time, decoded tokens (greedy for ``hf``,
speculative otherwise, plus a greedy run for the per-token decode cost), peak
working set and private memory of the process, VRAM in use, model size on disk,
character error against the ground truth, and parity with the BF16 reference
dumps of ``build/parity`` (``tools/ref_dump.py``): last prefill logits (relative
error, argmax) and the greedy token ids.  The table goes to the console and to
``<out>/table.md`` / ``table.json`` (``docs/BENCHMARK.md`` quotes it with the setup).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BACKENDS = ["hf", "numpy", "cpu", "rocm", "vulkan", "rocm@orig"]
LABELS = {
    "hf": "① original (transformers, CPU f32)",
    "numpy": "② Python / numpy (quantised)",
    "cpu": "③ Rust CPU kernels (quantised)",
    "rocm": "④ GPU ROCm (HIP kernels, quantised)",
    "vulkan": "④' GPU Vulkan (portable, quantised)",
    "rocm@orig": "④ GPU ROCm (HIP kernels, original BF16 weights)",
}


def pick_samples(sets: list[str], per: int) -> list[dict]:
    out = []
    for name in sets:
        folder = ROOT / "build" / "synth" / name
        seen: dict = {}
        for s in json.loads((folder / "samples.json").read_text(encoding="utf-8")):
            if seen.get(s["scenario"], 0) < per:
                seen[s["scenario"]] = seen.get(s["scenario"], 0) + 1
                out.append({**s, "set": name, "path": str(folder / s["image"])})
    return out


# ------------------------------------------------------------------------- workers
def prepared(sample: dict):
    from gravity_ocr.drawing import crops

    prep = crops.prepare_crop(Image.open(sample["path"]).convert("RGB"), prompt=sample["prompt"], text_px=30)
    return prep.image, prep.prompt


def worker_hf(args, samples, cases) -> dict:
    import torch
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration

    from gravity_ocr import memory, score

    torch.set_num_threads(os.cpu_count() // 2 or 1)
    t0 = time.perf_counter()
    proc = AutoProcessor.from_pretrained(ROOT / "models")
    model = GlmOcrForConditionalGeneration.from_pretrained(ROOT / "models", dtype=torch.float32).eval()
    load = time.perf_counter() - t0
    rows = []
    for s in samples:
        img, prompt = prepared(s)
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(img)}, {"type": "text", "text": prompt}]}]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        with torch.no_grad():
            t1 = time.perf_counter()
            model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            t_vis = time.perf_counter() - t1
            t1 = time.perf_counter()
            model(**inputs)
            t_pre = time.perf_counter() - t1 - t_vis
            t1 = time.perf_counter()
            gen = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
            t_gen = time.perf_counter() - t1
        new = gen[0, inputs["input_ids"].shape[1] :]
        text = proc.decode(new, skip_special_tokens=True)
        n = int(new.shape[0])
        # generate() runs vision and prefill again before the first token
        rows.append({"id": s["id"], "scenario": s["scenario"], "text": text, "tokens": n, "vision": t_vis, "prefill": max(t_pre, 0.0),
                     "decode": max(t_gen - t_vis - max(t_pre, 0.0), 1e-9), "greedy_decode": max(t_gen - t_vis - max(t_pre, 0.0), 1e-9),
                     "greedy_tokens": n, **score.score(text, s["gt_text"], s.get("gt_cells"))})  # fmt: skip
        print(f"  hf {s['id']} {n} tok vis {t_vis:.2f}s pre {t_pre:.2f}s gen {t_gen:.2f}s", flush=True)
    parity = []
    for case in cases:
        ref = np.load(ROOT / "build" / "parity" / "ref" / f"{case['id']}.npz")
        image = Image.open(ROOT / "build" / "parity" / case["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": case["prompt"]}]}]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].numpy()
            gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        ids = [int(t) for t in gen[0, inputs["input_ids"].shape[1] :]]
        want = [int(t) for t in ref["gen_ids"]]
        parity.append({"id": case["id"], "logits_rel": float(np.linalg.norm(logits - ref["prefill_logits_last"]) / np.linalg.norm(ref["prefill_logits_last"])),
                       "argmax_equal": int(logits.argmax()) == int(ref["prefill_logits_last"].argmax()), "ids_equal": ids == want})  # fmt: skip
    return {"load": load, "rows": rows, "parity": parity, "peak_ws": memory.peak(), "peak_private": memory.peak_private(),
            "model_bytes": (ROOT / "models" / "model.safetensors").stat().st_size, "device": "transformers-cpu-f32"}  # fmt: skip


def worker_runtime(args, samples, cases) -> dict:
    from gravity_ocr import decode, memory, score
    from gravity_ocr.model import GravityOCR

    device = {"numpy": "numpy", "cpu": "cpu", "rocm": "rocm", "vulkan": "vulkan"}[args.worker.split("@")[0]]
    vram0 = None
    if device in ("rocm", "vulkan"):
        from gravity_ocr import gpu

        lib = gpu.library()
        if lib is not None and device in lib.backends and lib.devices(device):
            probe = gpu.Engine(lib, device)
            vram0 = probe.mem()
            probe.close()
    t0 = time.perf_counter()
    model = GravityOCR.from_package(args.package, device=device) if args.package else GravityOCR.from_pretrained(ROOT / "models", device=device)
    load = time.perf_counter() - t0
    want = {"numpy": "numpy", "cpu": "cpu-native", "rocm": "gpu-rocm", "vulkan": "gpu-vulkan"}[device]
    if model.device != want:
        raise SystemExit(f"asked for {device}, got {model.device}")
    vram_peak = 0

    def vram():
        nonlocal vram_peak
        if model.gpu is not None and vram0 is not None:
            free, total = model.gpu.mem()
            vram_peak = max(vram_peak, vram0[0] - free)

    vram()
    rows = []
    for s in samples:
        img, prompt = prepared(s)
        out = decode.read(model, img, prompt, max_tokens=args.max_tokens)
        vram()
        g = decode.read(model, img, prompt, max_tokens=args.max_tokens, strategy="greedy") if args.greedy else None
        rows.append({"id": s["id"], "scenario": s["scenario"], "text": out.text, "tokens": len(out.tokens), "vision": out.stats["vision"],
                     "prefill": out.stats["prefill"], "decode": out.stats["decode"], "passes": out.stats["passes"],
                     "greedy_decode": g.stats["decode"] if g else None, "greedy_tokens": len(g.tokens) if g else None,
                     "greedy_equal": (g.tokens == out.tokens) if g else None,
                     **score.score(out.text, s["gt_text"], s.get("gt_cells"))})  # fmt: skip
        print(f"  {args.worker} {s['id']} {len(out.tokens)} tok vis {out.stats['vision']:.2f}s pre {out.stats['prefill']:.2f}s dec {out.stats['decode']:.2f}s", flush=True)
    parity = []
    from gravity_ocr import preprocess

    for case in cases:
        ref = np.load(ROOT / "build" / "parity" / "ref" / f"{case['id']}.npz")
        image = Image.open(ROOT / "build" / "parity" / case["image"]).convert("RGB")
        tokens, grid = model.encode_image(image, "resize")
        session = model.prefill(tokens, grid, case["prompt"])
        logits = session.logits
        ref_logits = ref["prefill_logits_last"]
        if model.head_ids is not None:
            ref_logits = ref_logits[model.head_ids]
        got = decode.greedy(model, session, max_tokens=256)
        want = [int(t) for t in ref["gen_ids"] if int(t) not in model.eos]
        parity.append({"id": case["id"], "logits_rel": float(np.linalg.norm(logits - ref_logits) / np.linalg.norm(ref_logits)),
                       "argmax_equal": decode.pick_of(model, logits, session.rest).token == int(np.argmax(ref["prefill_logits_last"])),
                       "ids_equal": got.tokens == want, "first_diff": next((i for i, (a, b) in enumerate(zip(got.tokens, want)) if a != b), None)})  # fmt: skip
        vram()
    del preprocess
    size = Path(args.package).stat().st_size if args.package else (ROOT / "models" / "model.safetensors").stat().st_size
    info = model.gpu.info if model.gpu is not None else {"threads": model.kernels.threads if model.kernels else 1}
    return {"load": load, "rows": rows, "parity": parity, "peak_ws": memory.peak(), "peak_private": memory.peak_private(),
            "vram_peak": vram_peak or None, "gpu_weights": model.gpu.weight_bytes if model.gpu is not None else None,
            "model_bytes": size, "device": model.device, "info": info}  # fmt: skip


# ------------------------------------------------------------------------- report
def summarise(name: str, res: dict) -> dict:
    rows = res["rows"]
    tok = sum(r["tokens"] for r in rows)
    dec = sum(r["decode"] for r in rows)
    gdec = sum(r["greedy_decode"] for r in rows if r.get("greedy_decode"))
    gtok = sum(r["greedy_tokens"] for r in rows if r.get("greedy_decode"))
    chars = sum(r["chars"] for r in rows)
    par = res.get("parity", [])
    return {
        "backend": LABELS.get(name, name), "device": res.get("device"), "load_s": round(res["load"], 2),
        "vision_ms": round(1e3 * np.mean([r["vision"] for r in rows]), 1), "prefill_ms": round(1e3 * np.mean([r["prefill"] for r in rows]), 1),
        "decode_ms_per_token": round(1e3 * gdec / max(gtok, 1), 2) if gtok else None, "tok_per_s": round(tok / max(dec, 1e-9), 1),
        "cer": round(sum(r["edits"] for r in rows) / max(chars, 1), 4), "exact": round(float(np.mean([r["edits_tight"] == 0 for r in rows])), 3),
        "peak_ws_mb": round(res["peak_ws"] / 2**20), "peak_private_mb": round(res["peak_private"] / 2**20),
        "vram_mb": round(res["vram_peak"] / 2**20) if res.get("vram_peak") else None,
        "model_mb": round(res["model_bytes"] / 2**20, 1),
        "parity_logits_rel": round(float(np.mean([p["logits_rel"] for p in par])), 5) if par else None,
        "parity_argmax": f"{sum(p['argmax_equal'] for p in par)}/{len(par)}" if par else None,
        "parity_ids": f"{sum(p['ids_equal'] for p in par)}/{len(par)}" if par else None,
        "samples": len(rows), "tokens": tok,
    }  # fmt: skip


def markdown(table: list[dict]) -> str:
    cols = [("backend", "backend"), ("load_s", "load s"), ("vision_ms", "vision ms"), ("prefill_ms", "prefill ms"),
            ("decode_ms_per_token", "decode ms/token (greedy)"), ("tok_per_s", "tok/s (as run)"), ("model_mb", "model MB"),
            ("peak_private_mb", "peak private MB"), ("peak_ws_mb", "peak working set MB"), ("vram_mb", "VRAM MB"),
            ("parity_logits_rel", "logits rel. err vs BF16"), ("parity_argmax", "argmax = BF16"), ("parity_ids", "greedy ids = BF16"),
            ("cer", "CER"), ("exact", "exact")]  # fmt: skip
    head = "| " + " | ".join(c[1] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join("" if r.get(k) is None else str(r[k]) for k, _ in cols) + " |\n" for r in table)
    return head + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--package", default="")
    ap.add_argument("--sets", default="heldout,heldout-dc")
    ap.add_argument("--per-scenario", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--greedy", action="store_true", default=True)
    ap.add_argument("--out", default=str(ROOT / "build" / "bench"))
    ap.add_argument("--worker", default="")
    args = ap.parse_args()
    samples = pick_samples(args.sets.split(","), args.per_scenario)
    cases = json.loads((ROOT / "build" / "parity" / "cases.json").read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        res = worker_hf(args, samples, cases) if args.worker == "hf" else worker_runtime(args, samples, cases)
        (out / f"{args.worker}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    table = []
    for b in args.backends.split(","):
        print(f"== {b}", flush=True)
        cmd = [sys.executable, "-X", "utf8", "-u", __file__, "--worker", b, "--sets", args.sets, "--per-scenario", str(args.per_scenario),
               "--max-tokens", str(args.max_tokens), "--out", str(out)]  # fmt: skip
        if args.package and b != "hf" and not b.endswith("@orig"):  # name@orig: the original weights on that backend
            cmd += ["--package", args.package]
        env = dict(os.environ)
        base = b.split("@")[0]
        if base in ("hf", "numpy"):
            env["GRAVITY_OCR_NATIVE"] = "0"  # pure numpy: no Rust kernel anywhere (pixels, decoders)
        if base in ("hf", "numpy", "cpu"):
            env["GRAVITY_OCR_GPU"] = "0"
        t0 = time.perf_counter()
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0 or not (out / f"{b}.json").exists():
            print(f"   {b} failed (exit {rc})", flush=True)
            continue
        res = json.loads((out / f"{b}.json").read_text(encoding="utf-8"))
        table.append(summarise(b, res))
        print(f"   done in {time.perf_counter() - t0:.0f}s: {table[-1]}", flush=True)
    md = markdown(table)
    print(md)
    (out / "table.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "table.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
