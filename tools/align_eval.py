"""``vision_align`` (plan Part 3.1): does an orthogonal Procrustes rotation bring the vision
tower's glyph features onto the language model's token rows?

    python tools/align_eval.py --device rocm

Single glyphs (digits, capitals, small letters, units' symbols, and the syllables of
the 232 whole-syllable Hangul tokens) are rendered in three fonts.  A glyph's vision
vector is the mean of the image tokens of its picture.  The fit uses the glyphs of
two fonts' "train" half; the test uses the other glyphs (never seen in the fit) in
every font.  Retrieval among all glyph tokens, top-1 / top-5:

(a) no alignment – the vision vector against the ``embed_tokens`` (and the ``lm_head``) rows
(b) Procrustes onto the ``lm_head`` rows      W = U Vᵀ,  U Σ Vᵀ = svd(Xᵀ Y + λ‖XᵀY‖ I)
(c) Procrustes onto the ``embed_tokens`` rows

λ = 1e-4 is the plan's; a sweep over λ (a large λ pulls W to the identity) shows
whether any strength beats no rotation at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gravity_ocr import language, preprocess, surgery  # noqa: E402
from gravity_ocr.model import GravityOCR  # noqa: E402
from gravity_ocr.synth import fonts  # noqa: E402

RIDGE = 1e-4


def glyphs(vocab) -> list[tuple[str, int]]:
    """``(character, token id)`` of every glyph that is one token on its own."""
    chars = [str(d) for d in range(10)] + [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)] + list("%℃μΩ±→·")
    for i in range(vocab.size):
        if not vocab.is_special(i) and surgery.token_class(vocab.piece(i)) == "hangul":
            s = vocab.piece(i).decode("utf-8")
            if len(s) == 1:
                chars.append(s)
    out = []
    for c in dict.fromkeys(chars):
        ids = vocab.encode(c)
        if len(ids) == 1:
            out.append((c, ids[0]))
    return out


def render(ch: str, role: str, size: int = 56) -> np.ndarray:
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = fonts.font(role, int(size * 0.7))
    x0, y0, x1, y1 = d.textbbox((0, 0), ch, font=f)
    d.text(((size - (x1 - x0)) / 2 - x0, (size - (y1 - y0)) / 2 - y0), ch, font=f, fill=(0, 0, 0))
    return np.asarray(img)


def procrustes(x: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """``U Vᵀ`` of ``svd(Xᵀ Y + ridge·‖XᵀY‖ I)``; a large ridge pulls the rotation to the identity."""
    c = x.T.astype(np.float64) @ y.astype(np.float64)
    c += ridge * np.linalg.norm(c) * np.eye(c.shape[0])
    u, _, vt = np.linalg.svd(c)
    return (u @ vt).astype(np.float32)


def l2(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def retrieval(q: np.ndarray, keys: np.ndarray, want: np.ndarray) -> tuple[float, float]:
    sims = l2(q) @ l2(keys).T
    order = np.argsort(-sims, axis=1)
    top1 = float(np.mean(order[:, 0] == want))
    top5 = float(np.mean([w in o[:5] for o, w in zip(order, want)]))
    return top1, top5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(ROOT / "build" / "eval" / "align.json"))
    args = ap.parse_args()
    model = GravityOCR.from_pretrained(ROOT / "models", device=args.device)
    gl = glyphs(model.vocab)
    roles = ["gothic", "bold", "hand"]
    feats = {}
    for role in roles:
        items = []
        for ch, _ in gl:
            patches, grid = preprocess.to_patches(preprocess.fit_image(render(ch, role)))
            items.append((patches, grid, None))
        if hasattr(model.vision, "forward_batch"):
            toks = []
            for i in range(0, len(items), 64):
                toks += model.vision.forward_batch(items[i : i + 64])
        else:
            toks = [model.vision.forward(p, g) for p, g, _ in items]
        feats[role] = np.stack([t.mean(axis=0) for t in toks])
    ids = np.array([t for _, t in gl])
    head = np.asarray(model.src.rows(language.HEAD, ids))
    emb = np.asarray(model.src.rows(language.EMBED, ids))
    rng = np.random.default_rng(0)
    train = rng.random(len(gl)) < 0.5
    x_tr = np.concatenate([feats[r][train] for r in roles[:2]])
    w_head = procrustes(l2(x_tr), l2(np.concatenate([head[train]] * 2)))
    w_emb = procrustes(l2(x_tr), l2(np.concatenate([emb[train]] * 2)))
    test = np.flatnonzero(~train)
    report = {"glyphs": len(gl), "train": int(train.sum()), "test": int(test.size)}
    for role in roles:
        q = feats[role][test]
        want = test  # index into the glyph list (keys are all glyphs)
        report[role] = {
            "a_none_vs_embed": retrieval(q, emb, want),
            "a_none_vs_head": retrieval(q, head, want),
            "b_procrustes_head": retrieval(q @ w_head, head, want),
            "c_procrustes_embed": retrieval(q @ w_emb, emb, want),
            "b_train_fit": retrieval(feats[role][train] @ w_head, head, np.flatnonzero(train)),
        }
        print(role, report[role], flush=True)
    # the ridge pulls the rotation towards the identity: does any strength beat no rotation?
    sweep = {}
    for ridge in (1e-4, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0):
        for name, keys in (("head", head), ("embed", emb)):
            w = procrustes(l2(x_tr), l2(np.concatenate([keys[train]] * 2)), ridge)
            sweep[f"{name}@{ridge:g}"] = [round(float(np.mean([retrieval(feats[r][test] @ w, keys, test)[k] for r in roles])), 4) for k in (0, 1)]
    report["ridge_sweep_top1_top5_mean_over_fonts"] = sweep
    print("ridge sweep", sweep, flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
