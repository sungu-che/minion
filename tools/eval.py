"""Read every crop of a synthetic set and score it against its ground truth.

    python tools/eval.py --data build/synth/calib --name f32-up30 --text-px 30

One run is one way of showing the crops to one model; the knobs are the ones
``prepare_crop`` has.  The report goes to ``<out>/<set>-<name>.json`` and a
table per scenario to the console.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gravity_ocr import decode, score  # noqa: E402
from gravity_ocr.drawing import crops  # noqa: E402
from gravity_ocr.model import TABLE_PROMPT, TEXT_PROMPT, GravityOCR  # noqa: E402

PROMPTS = {"text": TEXT_PROMPT, "table": TABLE_PROMPT}


def load_model(args) -> GravityOCR:
    if getattr(args, "gguf", None):
        return GravityOCR.from_gguf(args.gguf, device=args.device)
    if args.package:
        return GravityOCR.from_package(args.package, device=args.device)
    return GravityOCR.from_pretrained(args.models, device=args.device)


def head_slice(model: GravityOCR, budget: int):
    """The output slice built from the calibration ground truth only (never from what is scored)."""
    from gravity_ocr import surgery

    texts = []
    for name in ("calib", "calib-dc"):
        path = Path("build/synth") / name / "samples.json"
        if path.exists():
            texts += [s["gt_text"] for s in json.loads(path.read_text(encoding="utf-8"))]
    return surgery.build(model.vocab, surgery.domain_corpus(texts), budget)


def show(image, sample: dict, args) -> tuple:
    """The image and prompt one sample is read with, and what the crop analysis said."""
    want = PROMPTS[sample["prompt"]]
    if args.prep == "none":
        return image, want, None
    prep = crops.prepare_crop(
        image,
        prompt=None if args.prompt == "auto" else sample["prompt"],
        text_px=args.text_px or None,
        damp=args.damp,
        grid=args.grid,
        grid_frame=args.grid_frame,
        level=args.level,
    )
    return prep.image, prep.prompt, prep.info.summary()


def decoded(model: GravityOCR, samples: list[dict], root: Path, args):
    """``(sample, prompt, info, prep seconds, Decoded)`` for every sample, ``--batch`` crops at a time."""
    step = max(args.batch, 1)
    for i in range(0, len(samples), step):
        chunk = samples[i : i + step]
        shown = []
        for s in chunk:
            image = Image.open(root / s["image"]).convert("RGB")
            t0 = time.perf_counter()
            img, prompt, info = show(image, s, args)
            shown.append((img, prompt, info, time.perf_counter() - t0))
        la = getattr(args, "_lookahead", None)
        if args.batch > 1:
            outs = decode.read_many(model, [(img, prompt) for img, prompt, _, _ in shown], fit=args.fit, strategy=args.strategy,
                                    max_tokens=args.max_tokens, prune=args.prune, lookahead=la, batch_patches=args.batch_patches)  # fmt: skip
        else:
            outs = [decode.read(model, img, prompt, fit=args.fit, strategy=args.strategy, max_tokens=args.max_tokens, prune=args.prune,
                                lookahead=la) for img, prompt, _, _ in shown]  # fmt: skip
        for s, (_, prompt, info, t_prep), out in zip(chunk, shown, outs):
            yield s, prompt, info, t_prep, out


def run(model: GravityOCR, samples: list[dict], root: Path, args, log=print) -> dict:
    rows = []
    for s, prompt, info, t_prep, out in decoded(model, samples, root, args):
        rec = score.score(out.text, s["gt_text"], s.get("gt_cells"))
        rec.update(
            id=s["id"], scenario=s["scenario"], prompt=prompt, prompt_ok=prompt == PROMPTS[sample_prompt(s)],
            text=out.text, tokens=out.tokens, conf=[round(p.conf, 4) for p in out.picks],
            alts=[[i, [[t, round(pr, 4)] for t, pr in p.alts]] for i, p in enumerate(out.picks) if p.alts],
            low=sum(1 for p in out.picks if p.conf < decode.SHADOW_CEIL), prep=round(t_prep, 4), info=info,
            stats={k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.stats.items()},
        )  # fmt: skip
        rows.append(rec)
        log(
            f"{s['id']:<18} cer={rec['edits'] / max(rec['chars'], 1):.3f} tok={len(out.tokens):>3} "
            f"pass={out.stats['passes']:>3} v={out.stats['vision']:.2f}s p={out.stats['prefill']:.2f}s "
            f"d={out.stats['decode']:.2f}s | {out.text[:60]!r}"
        )
    return {"rows": rows, "summary": summary(rows)}


def sample_prompt(s: dict) -> str:
    return s["prompt"]


def summary(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["scenario"]].append(r)
    out = {k: score.summarise(v) for k, v in by.items()}
    out["all"] = score.summarise(rows)
    for k, v in list(by.items()) + [("all", rows)]:
        toks = sum(len(r["tokens"]) for r in v)
        dec = sum(r["stats"]["decode"] for r in v)
        out[k].update(
            tokens=toks,
            passes=sum(r["stats"]["passes"] for r in v),
            sec_prep=round(sum(r["prep"] for r in v), 3),
            sec_vision=round(sum(r["stats"]["vision"] for r in v), 3),
            sec_prefill=round(sum(r["stats"]["prefill"] for r in v), 3),
            sec_decode=round(dec, 3),
            tok_per_s=round(toks / dec, 2) if dec else 0.0,
            prompt_ok=sum(1 for r in v if r["prompt_ok"]),
            low_conf=sum(r["low"] for r in v),
            rescued=sum(r["stats"].get("rescued", 0) for r in v),
        )
    return out


def table(summ: dict) -> str:
    head = f"{'scenario':<10} {'n':>3} {'CER':>7} {'tight':>7} {'exact':>6} {'cells':>6} {'shift':>5} {'tok':>5} {'pass':>5} {'tok/s':>6} {'vis s':>6} {'pre s':>6} {'dec s':>6}"
    lines = [head, "-" * len(head)]
    for k, v in summ.items():
        cells = f"{v['cell_exact']:.3f}" if "cell_exact" in v else "-"
        shift = str(v["shifted_rows"]) if "shifted_rows" in v else "-"
        lines.append(
            f"{k:<10} {v['n']:>3} {v['cer']:>7.4f} {v['cer_tight']:>7.4f} {v['exact']:>6.3f} {cells:>6} {shift:>5} "
            f"{v['tokens']:>5} {v['passes']:>5} {v['tok_per_s']:>6.1f} {v['sec_vision']:>6.1f} {v['sec_prefill']:>6.1f} {v['sec_decode']:>6.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--package", default=None)
    ap.add_argument("--gguf", default=None, help="GGUF pyramid model (tools/build_gguf.py); GPU only")
    ap.add_argument("--device", default="auto", help="auto | gpu | rocm | vulkan | cuda | cpu | numpy")
    ap.add_argument("--slice-head", type=int, default=0, help="keep this many output rows (surgery.build on calib text)")
    ap.add_argument("--rescue", action="store_true", help="with --slice-head: check the cut rows every pass (a package carries them)")
    ap.add_argument("--data", default="build/synth/calib")
    ap.add_argument("--out", default="build/eval")
    ap.add_argument("--name", required=True)
    ap.add_argument("--prep", choices=("none", "crop"), default="crop")
    ap.add_argument("--text-px", type=float, default=crops.TEXT_PX, help="0 keeps the crop size")
    ap.add_argument("--damp", action="store_true", help="damp lines through text (off by default, see crops.py)")
    ap.add_argument("--no-grid", dest="grid", action="store_false")
    ap.add_argument("--grid-frame", action="store_true")
    ap.add_argument("--level", action="store_true")
    ap.add_argument("--prompt", choices=("gt", "auto"), default="gt")
    ap.add_argument("--fit", choices=("pad", "resize"), default="pad")
    ap.add_argument("--strategy", choices=("speculative", "greedy"), default="speculative")
    ap.add_argument("--prune", choices=("off", "vision", "prefill", "vq"), default="off", help="blank-patch pruning (prune.py)")
    ap.add_argument("--vq", default="", help="VQ codebook package (tools/vq_build.py) for --prune vq")
    ap.add_argument("--batch", type=int, default=1, help="crops read together (their vision passes are batched on a GPU)")
    ap.add_argument("--batch-patches", type=int, default=4_096, help="patch rows one batched vision pass may take")
    ap.add_argument("--lookahead", action="store_true", help="draft look-ahead from the domain seeds and the set's ledger codes")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--scenarios", default=None, help="comma separated; default all")
    ap.add_argument("--limit", type=int, default=0, help="first N samples of every scenario")
    args = ap.parse_args()

    root = Path(args.data)
    samples = json.loads((root / "samples.json").read_text(encoding="utf-8"))
    if args.scenarios:
        keep = set(args.scenarios.split(","))
        samples = [s for s in samples if s["scenario"] in keep]
    if args.limit:
        seen: dict = defaultdict(int)
        picked = []
        for s in samples:
            seen[s["scenario"]] += 1
            if seen[s["scenario"]] <= args.limit:
                picked.append(s)
        samples = picked
    t0 = time.perf_counter()
    model = load_model(args)
    if args.slice_head:
        model.slice_head(head_slice(model, args.slice_head).ids, rescue=args.rescue)
    if args.vq:
        from gravity_ocr import vq as vqmod
        from gravity_ocr.container import Package

        model.vq = vqmod.Codebook.load(Package(args.vq))
    if args.lookahead:
        from gravity_ocr import ngram

        led = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
        codes = [e["kks"] for e in led["equipment"]] + [s["tag"] for s in led["sensors"]]
        args._lookahead = ngram.lookahead(model.vocab, ngram.seed_texts(codes) + ngram.code_texts(codes))
    print(f"model loaded on {model.device} in {time.perf_counter() - t0:.1f}s; {len(samples)} samples from {root}", flush=True)
    t0 = time.perf_counter()
    report = run(model, samples, root, args, log=lambda m: print(m, flush=True))
    report["args"] = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    report["device"] = model.device
    report["seconds"] = round(time.perf_counter() - t0, 1)
    if model.residency is not None:  # what was on the device, in which precision, and the swaps
        hist = model.residency.history
        report["memory"] = model.memory
        report["memory"]["loads"] = sum(1 for t in hist if t.action == "load")
        report["memory"]["load_seconds"] = round(sum(t.seconds for t in hist if t.action == "load"), 2)
        report["memory"]["stores"] = sorted({f"{t.group}:{t.store}" for t in hist if t.action == "load"})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{root.name}-{args.name}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n== {root.name} / {args.name}  ({report['seconds']}s)")
    print(table(report["summary"]))
    if "memory" in report:
        mem = report["memory"]
        print(f"memory: {mem['policy']} {' '.join(mem['stores'])} kv={mem['kv']} loads={mem['loads']} "
              f"({mem['load_seconds']}s) free {mem['free_mb']:.0f}/{mem['total_mb']:.0f} MB")  # fmt: skip
    print(f"report: {path}", flush=True)


if __name__ == "__main__":
    main()
