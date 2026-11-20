#!/usr/bin/env python3
"""Erase text that sits over art (a character, screentone, speed lines) with Nano Banana Pro, one crop per
text block, changing pixels ONLY on the block's letters.

As a module: `erase_art_text(base, art_mask, blocks, work, cost) -> (image, generations)`
             `confident_blocks(det_json, raw_json, keep_x) -> blocks`.

Why crops (2026-11-20, page 080): one whole-page erase (gpt_image_2_5 low, 2k) came back as
grey smudges with no drawing under the letters. A crop around the block reaches the model at
full detail, so it redraws the lines and screentone behind the letters.

Per block:
  1. confidence: the detector scans the page in overlapping tiles, so real text is found many
     times (080: 15-40 raw boxes per block) and a false hit - a rock in screentone - once.
     A block needs MIN_VOTES raw boxes mostly inside it. Blocks entirely outside `keep_x`
     (the neighbouring page's cut margin, which is cut away later) are skipped.
  2. crop: the block plus PAD_FRAC of its size (min PAD_MIN px), widened to the model aspect
     ratio closest to it, uploaded with its long side at 2048 px.
  3. the answer is registered onto the crop (affine ECC outside the block), tone-matched on a
     ring around the letters, and used only on the letters (art mask, grown by the caller),
     feathered. Everything else stays the scan.

Example: see add_clean_layer.py, which calls it.
"""
import json
import os
from typing import List, Tuple

import cv2
import numpy as np

import gutter_ai

MIN_VOTES = 3            # raw detections a block needs to count as text
PAD_FRAC = 0.6           # context around a block, share of its larger side
PAD_MIN = 48             # px
FEATHER = 2.0            # px soft edge where the erase meets the scan
ASPECTS = {"1:1": 1.0, "3:2": 1.5, "2:3": 2 / 3, "4:3": 4 / 3, "3:4": 0.75, "4:5": 0.8, "5:4": 1.25,
           "9:16": 9 / 16, "16:9": 16 / 9, "21:9": 21 / 9}
TEXT_PROMPT = (
    "This is a crop of a scanned black-and-white manga page. Erase ONLY the lettering in it - every "
    "letter, character, number and punctuation mark, in any script - and rebuild what was behind the "
    "letters as a seamless continuation of the drawing: the same lines, hatching, screentone dots, "
    "texture and grey tones as around them. Leave no ghost strokes, no smudges, no blur and never write "
    "new text. Do not change anything else: same composition, every line where it is, same framing. "
    "Black and white only.")
Block = Tuple[int, int, int, int]


def _votes(block: Block, raw: List[List[int]]) -> int:
    """Raw boxes with at least half their area inside the block."""
    x, y, w, h = block
    n = 0
    for rx, ry, rw, rh in raw:
        ix = max(0, min(x + w, rx + rw) - max(x, rx))
        iy = max(0, min(y + h, ry + rh) - max(y, ry))
        n += ix * iy >= 0.5 * max(1, rw * rh)
    return n


def confident_blocks(det_json: str, raw_json: str, keep_x: Tuple[int, int]) -> List[Block]:
    """Text blocks found MIN_VOTES+ times that reach into x range keep_x. Example: 080 -> 1 block of 4."""
    blocks = json.load(open(det_json))["text_blocks"]
    raw = json.load(open(raw_json)) if os.path.exists(raw_json) else []
    return [tuple(b) for b in blocks
            if _votes(b, raw) >= MIN_VOTES and b[0] + b[2] > keep_x[0] and b[0] < keep_x[1]]


def crop_box(block: Block, shape: Tuple[int, int]) -> Tuple[int, int, int, int, str]:
    """(x0, y0, w, h, aspect) of the context crop around a block, inside the page."""
    x, y, w, h = block
    pad = max(PAD_MIN, int(PAD_FRAC * max(w, h)))
    cw, ch = w + 2 * pad, h + 2 * pad
    aspect = min(ASPECTS, key=lambda k: abs(np.log((cw / ch) / ASPECTS[k])))
    r = ASPECTS[aspect]
    cw, ch = (int(round(ch * r)), ch) if cw / ch < r else (cw, int(round(cw / r)))
    H, W = shape
    cw, ch = min(cw, W), min(ch, H)
    x0 = int(np.clip(x + w / 2 - cw / 2, 0, W - cw))
    y0 = int(np.clip(y + h / 2 - ch / 2, 0, H - ch))
    return x0, y0, cw, ch, aspect


def upload_size(aspect: str) -> Tuple[int, int]:
    r = ASPECTS[aspect]
    return (2048, int(round(2048 / r))) if r >= 1 else (int(round(2048 * r)), 2048)


def erase_art_text(base: np.ndarray, art: np.ndarray, blocks: List[Block], work: str,
                   cost: bool = False) -> Tuple[np.ndarray, int]:
    """(base with the letters over art erased, generations). `art`: 255 on the letters (already grown)."""
    os.makedirs(work, exist_ok=True)
    out = base.astype(np.float32)
    n = 0
    for i, b in enumerate(blocks):
        x0, y0, cw, ch, aspect = crop_box(b, base.shape[:2])
        ys, xs = slice(y0, y0 + ch), slice(x0, x0 + cw)
        letters = np.zeros(art.shape, np.uint8)
        x, y, w, h = b
        letters[y:y + h, x:x + w] = art[y:y + h, x:x + w]
        if not letters.any():
            continue
        crop = base[ys, xs]
        n += 1
        gen = gutter_ai.generate(crop, os.path.join(work, f"crop_{i}.png"), cost, TEXT_PROMPT, aspect, upload_size(aspect))
        if cost:
            continue
        lm = letters[ys, xs]
        keep = np.ones(lm.shape, bool)
        keep[max(0, y - y0 - 8):y - y0 + h + 8, max(0, x - x0 - 8):x - x0 + w + 8] = False
        reg = gutter_ai.match_tones(gutter_ai.register(gen, crop, keep), crop, lm)
        soft = np.maximum(cv2.GaussianBlur((lm > 0).astype(np.float32), (0, 0), FEATHER), (lm > 0).astype(np.float32))
        out[ys, xs] = out[ys, xs] * (1 - soft[..., None]) + reg * soft[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), n
