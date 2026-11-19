"""Font-size-aware downscale for large *text* crops (Track 2 speed path; tables and formulas are never touched).

GLM-OCR's processor keeps every crop at native resolution up to 4.8 MP, so a magazine or exam-paper paragraph scanned
at high DPI (line pitch 90-190 px) costs up to 24k ViT patches, and the ViT's full attention is quadratic in that. Such
crops carry the same text at a fraction of the resolution: the model reads documents whose lines are ~25-40 px apart.

text_downscale() measures the text-line pitch from the horizontal ink profile and returns a scale that brings the pitch
down to TARGET_PITCH (64 px by default, ~2x a normal document scan), never below MIN_SCALE, and only when the line
structure is unambiguous (>= 3 regular lines, or 1-2 clearly separated lines). Anything else -> 1.0 (unchanged).

Env: TRACK2_TEXT_DOWNSCALE (1 = on), TRACK2_TEXT_PITCH (target px), TRACK2_TEXT_MIN_PATCHES (only crops above it).
"""
from __future__ import annotations

import math
import os
from typing import Tuple

import numpy as np

ENABLED = os.environ.get("TRACK2_TEXT_DOWNSCALE", "1") == "1"
TARGET_PITCH = float(os.environ.get("TRACK2_TEXT_PITCH", "64"))
MIN_PATCHES = int(os.environ.get("TRACK2_TEXT_MIN_PATCHES", "4096"))
MIN_SCALE = float(os.environ.get("TRACK2_TEXT_MIN_SCALE", "0.34"))
FACTOR = 28


def _patches(h: int, w: int) -> int:
    # smart_resize's grid (no budget clamp needed for the gate)
    hb = max(FACTOR, round(h / FACTOR) * FACTOR)
    wb = max(FACTOR, round(w / FACTOR) * FACTOR)
    return (hb // 14) * (wb // 14)


def _gray(img) -> np.ndarray:
    """uint8 gray from HWC numpy or CHW torch uint8; columns subsampled to <= ~1024 for speed."""
    if hasattr(img, "numpy"):  # torch CHW
        a = img.numpy()
        h, w = a.shape[1], a.shape[2]
        step = max(1, w // 1024)
        g = a[1, :, ::step]
    else:
        a = np.asarray(img)
        h, w = a.shape[0], a.shape[1]
        step = max(1, w // 1024)
        g = a[:, ::step, 1] if a.ndim == 3 else a[:, ::step]
    return np.ascontiguousarray(g), h, w


def text_downscale(img, task: str) -> Tuple[float, str]:
    """-> (scale <= 1.0, short diagnostic string)."""
    if not ENABLED or task != "text":
        return 1.0, "off"
    g, h, w = _gray(img)
    if _patches(h, w) < MIN_PATCHES:
        return 1.0, "small"
    paper = float(np.percentile(g, 90))
    if paper < 150:  # dark / inverted / photo background: no reliable ink profile
        return 1.0, f"paper{paper:.0f}"
    ink = g < (paper - 70)
    cover = float(ink.mean())
    if not (0.01 <= cover <= 0.40):
        return 1.0, f"cover{cover:.3f}"
    row = ink.mean(axis=1)
    thr = max(0.004, 0.08 * float(np.percentile(row, 95)))
    on = row > thr
    # runs of ink rows (text lines)
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    hts = ends - starts
    keep = hts >= 4
    starts, ends, hts = starts[keep], ends[keep], hts[keep]
    n = int(len(starts))
    if n == 0:
        return 1.0, "nolines"
    if n >= 3:
        pitches = np.diff(starts).astype(np.float64)
        pitch = float(np.median(pitches))
        cv = float(np.std(pitches) / max(pitch, 1.0))
        lh = float(np.median(hts))
        if cv > 0.35 or lh > 0.85 * pitch:
            return 1.0, f"irregular n={n} pitch={pitch:.0f} cv={cv:.2f} lh={lh:.0f}"
        eff = pitch
    else:
        lh = float(np.max(hts))
        if lh > 0.45 * h:  # one run covering most of the crop: lines not separable
            return 1.0, f"merged n={n} lh={lh:.0f}"
        eff = 1.35 * lh
    scale = max(MIN_SCALE, min(1.0, TARGET_PITCH / eff))
    if scale > 0.9:
        return 1.0, f"fine n={n} pitch={eff:.0f}"
    return scale, f"n={n} pitch={eff:.0f}"


def scaled_size(h: int, w: int, scale: float) -> Tuple[int, int]:
    return max(1, int(math.floor(h * scale + 0.5))), max(1, int(math.floor(w * scale + 0.5)))
