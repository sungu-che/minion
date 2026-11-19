"""Compare the runtime with the reference dump written by ``ref_dump.py``.

    python tools/parity.py --ref build/parity/ref --cases build/parity/cases.json --device rocm

``--device`` picks the backend (numpy, the native CPU kernels, a GPU engine);
``--package`` checks a ``.gocr`` package (quantised, sliced head) instead of the
original weights.  Hidden states are compared on CPU devices only (the GPU
engine does not hand them out); with a sliced head the logits are compared on
the kept rows.

Each stage is fed the reference's own input so an error cannot hide behind an
earlier one: patches → image tokens, image tokens → hidden states → logits,
then the decoded ids.  The last block runs the runtime end to end from the PNG.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gravity_ocr import decode, preprocess  # noqa: E402
from gravity_ocr.model import GravityOCR  # noqa: E402


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--package", default=None, help="check a .gocr package instead of the original weights")
    ap.add_argument("--device", default="numpy", help="numpy | cpu | rocm | vulkan | cuda | gpu | auto")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.package:
        model = GravityOCR.from_package(args.package, device=args.device)
    else:
        model = GravityOCR.from_pretrained(args.models, device=args.device)
    print(f"runtime loaded on {model.device} in {time.perf_counter() - t0:.1f}s")
    on_cpu = model.gpu is None
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    base = Path(args.cases).parent
    report = []
    for case in cases:
        ref = np.load(Path(args.ref) / f"{case['id']}.npz")
        _, gh, gw = (int(v) for v in ref["image_grid_thw"][0])
        grid = (gh // 2, gw // 2)
        row: dict = {"id": case["id"], "grid": [gh, gw]}

        # 1. prompt
        ids = model.prompt_ids(case["prompt"], grid[0] * grid[1])
        row["prompt_ids_equal"] = ids == ref["input_ids"].tolist()

        # 2. preprocessing (the reference resizes with torchvision, the runtime with Pillow)
        image = Image.open(base / case["image"]).convert("RGB")
        mine, g2 = preprocess.to_patches(preprocess.fit_image(image, "resize"))
        theirs = preprocess.fold_reference_patches(ref["pixel_values"])
        row["resized"] = (image.height, image.width) != (gh * 14, gw * 14)
        row["grid_equal"] = g2 == (gh, gw)
        if row["grid_equal"]:
            row["pixel_max_abs"] = float(np.max(np.abs(mine - theirs)))
            row["pixel_mean_abs"] = float(np.mean(np.abs(mine - theirs)))

        # 3. vision on the reference patches
        t1 = time.perf_counter()
        tokens = model.vision.forward(theirs, (gh, gw))
        row["sec_vision"] = round(time.perf_counter() - t1, 3)
        row["image_tokens_rel"] = rel(tokens, ref["image_embeds"])
        row["image_tokens_max_abs"] = float(np.max(np.abs(tokens - ref["image_embeds"])))

        # 4. language model on the reference image tokens
        hidden: list | None = [] if on_cpu else None
        t1 = time.perf_counter()
        session = model.prefill(ref["image_embeds"], grid, case["prompt"], hidden=hidden)
        row["sec_prefill"] = round(time.perf_counter() - t1, 3)
        if hidden and "hidden_states" in ref.files:
            hs = ref["hidden_states"]
            row["hidden_rel"] = [round(rel(h, hs[i]), 7) for i, h in enumerate(hidden)]
        ref_logits = ref["prefill_logits_last"]
        if model.head_ids is not None:
            ref_logits = ref_logits[model.head_ids]
        row["logits_rel"] = rel(session.logits, ref_logits)
        row["logits_max_abs"] = float(np.max(np.abs(session.logits - ref_logits)))
        # the runtime's pick (a sliced head checks its cut rows, see GravityOCR.set_rest_parts)
        row["argmax_equal"] = decode.pick_of(model, session.logits, session.rest).token == int(np.argmax(ref["prefill_logits_last"]))

        # 5. greedy decoding from the reference image tokens
        want = [int(t) for t in ref["gen_ids"] if int(t) not in model.eos]
        t1 = time.perf_counter()
        got = decode.greedy(model, session, max_tokens=256)
        row["sec_greedy"] = round(time.perf_counter() - t1, 3)
        row["greedy_ids_equal"] = got.tokens == want
        row["text"] = got.text
        scores = ref["gen_scores"]
        ref_conf = [float(np.exp(s.max() - np.logaddexp.reduce(s.astype(np.float64)))) for s in scores[: len(want)]]
        row["conf_max_abs"] = float(max(abs(a.conf - b) for a, b in zip(got.picks, ref_conf))) if want else 0.0

        # 6. speculative decoding must give the same ids
        session = model.prefill(ref["image_embeds"], grid, case["prompt"])
        spec = decode.speculative(model, session, max_tokens=256, early_exit=None)
        row["speculative_ids_equal"] = spec.tokens == want
        row["speculative"] = {k: spec.stats[k] for k in ("passes", "rows", "drafted", "accepted", "seconds")}
        session = model.prefill(ref["image_embeds"], grid, case["prompt"])
        spec = decode.speculative(model, session, max_tokens=256)
        row["early_exit_ids_equal"] = spec.tokens == want
        row["early_exit"] = {k: spec.stats[k] for k in ("passes", "rows", "early_exit", "seconds")}

        # 7. end to end from the PNG, both ways of fitting the grid
        for fit in ("resize", "pad"):
            tokens, g = model.encode_image(image, fit)
            out = decode.greedy(model, model.prefill(tokens, g, case["prompt"]), max_tokens=256)
            row[f"e2e_{fit}"] = {"grid": list(g), "ids_equal": out.tokens == want, "text": out.text}
        report.append(row)
        print(json.dumps(row, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
