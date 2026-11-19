"""Reference oracle: run the official ``transformers`` implementation and dump tensors.

Development only.  It needs ``torch`` and ``transformers``; the runtime in
``gravity_ocr`` never imports either.  The dump is what the numpy runtime is
compared against (``tests/test_parity.py``, ``tools/parity.py``).

    python tools/ref_dump.py --cases build/parity/cases.json --out build/parity/ref
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, GlmOcrForConditionalGeneration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--cases", required=True, help="json list of {id, image, prompt}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--hidden", action="store_true", help="also dump per-layer hidden states")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    base = Path(args.cases).parent

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.models)
    model = GlmOcrForConditionalGeneration.from_pretrained(args.models, dtype=torch.float32)
    model.eval()
    print(f"loaded in {time.time() - t0:.1f}s, image processor = {type(processor.image_processor).__name__}")

    summary = []
    for case in cases:
        image = Image.open(base / case["image"]).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": case["prompt"]}],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        dump: dict[str, np.ndarray] = {
            "input_ids": inputs["input_ids"][0].numpy(),
            "pixel_values": inputs["pixel_values"].numpy(),
            "image_grid_thw": inputs["image_grid_thw"].numpy(),
        }
        with torch.no_grad():
            t1 = time.time()
            feats = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            embeds = feats.pooler_output
            if isinstance(embeds, (list, tuple)):
                embeds = torch.cat(list(embeds), dim=0)
            dump["image_embeds"] = embeds.numpy()
            t_vision = time.time() - t1

            t1 = time.time()
            fwd = model(**inputs, output_hidden_states=args.hidden)
            t_prefill = time.time() - t1
            dump["prefill_logits_last"] = fwd.logits[0, -1].numpy()
            dump["prefill_logits_tail"] = fwd.logits[0, -8:].numpy()
            if args.hidden:
                dump["hidden_states"] = torch.stack([h[0] for h in fwd.hidden_states]).numpy()

            t1 = time.time()
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
            t_gen = time.time() - t1
        new_ids = gen.sequences[0, inputs["input_ids"].shape[1] :]
        dump["gen_ids"] = new_ids.numpy()
        dump["gen_scores"] = torch.stack([s[0] for s in gen.scores]).numpy().astype(np.float32)
        text = processor.decode(new_ids, skip_special_tokens=True)
        np.savez(out / f"{case['id']}.npz", **dump)
        row = {
            "id": case["id"],
            "prompt": case["prompt"],
            "prompt_text": processor.decode(inputs["input_ids"][0][:4])
            + " ... "
            + processor.decode(inputs["input_ids"][0][-12:]),
            "grid": dump["image_grid_thw"][0].tolist(),
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "new_tokens": int(new_ids.shape[0]),
            "text": text,
            "sec_vision": round(t_vision, 3),
            "sec_prefill": round(t_prefill, 3),
            "sec_generate": round(t_gen, 3),
        }
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False))

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
