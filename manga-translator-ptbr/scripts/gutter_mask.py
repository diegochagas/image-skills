#!/usr/bin/env python3
"""Find the middle line of a double-page scan and the scan defects around it.

    gutter_mask.py <image> [<image> ...] [--out DIR]

For each image writes <out>/<stem>_gutter_mask.png (255 = gutter + defects) and
<stem>_gutter_overlay.jpg (the page with the mask in green), and prints the
band found. As a module: `gutter_mask(img_bgr) -> (mask uint8, info dict)`.

How: the seam where the two pages meet is a straight, nearly vertical step
(paper tone changes across the fold) that shows in almost every height chunk
of the page. Each chunk is averaged over its rows, which cancels screentone
and drawn lines; a tilt search lines the chunks up for a seam that is not
exactly vertical. The best (tilt, column) is the seam; the paper shadow
that darkens the columns next to it is added on both sides, with a safety
margin. Returns an empty mask when no seam shows (SEAM_MIN_SCORE).

Example: gutter_mask.py 072-073.png --out /tmp/gutter
"""
import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SEARCH_FRAC = 0.20       # look for the seam within +- this fraction of the width from the centre
CHUNKS = 20              # the page height is cut in this many chunks
STEP_MIN = 12            # grey-level step across a seam, in a chunk's row average
SEAM_MIN_SCORE = 0.45    # share of chunks that must show the step along the tilted line
TILT_MAX_FRAC = 0.022    # largest seam tilt (x shift over the whole height) as a fraction of the width
EDGE_MARGIN = 40         # columns of the window ignored at its ends
PAD_FRAC = 0.012         # safety margin around the seam, fraction of the width
SHADOW_DROP = 0.06       # a column whose paper level is this much below the page's is shadow
SHADOW_MAX_FRAC = 0.05   # the shadow is followed at most this far (fraction of the width)
Seam = Tuple[float, float, float]    # (x at mid height, tilt over the height, score)


def _chunk_steps(gray: np.ndarray, x0: int, x1: int) -> np.ndarray:
    """(CHUNKS x window) grey step across each column: mean of 5 px on its right minus 5 px on its left."""
    h = gray.shape[0]
    rows = h // CHUNKS
    chunks = gray[:rows * CHUNKS].astype(np.float32).reshape(CHUNKS, rows, -1).mean(axis=1)
    cs = np.cumsum(np.pad(chunks, ((0, 0), (1, 0))), axis=1)
    x = np.arange(x0, x1)
    return (cs[:, x + 6] - cs[:, x + 1]) / 5 - (cs[:, x] - cs[:, x - 5]) / 5


def find_seam(gray: np.ndarray) -> Optional[Seam]:
    """Best straight seam near the centre: (x at mid height, tilt, score) or None."""
    w = gray.shape[1]
    x0, x1 = int(w * (0.5 - SEARCH_FRAC)), int(w * (0.5 + SEARCH_FRAC))
    steps = _chunk_steps(gray, x0, x1)
    tilt_max = int(w * TILT_MAX_FRAC)
    best: Optional[Seam] = None
    for tilt in range(-tilt_max, tilt_max + 1, 8):
        shifts = [int(round(tilt * (i / (CHUNKS - 1) - 0.5))) for i in range(CHUNKS)]
        aligned = np.stack([np.roll(steps[i], -shifts[i]) for i in range(CHUNKS)])
        score = np.maximum((aligned > STEP_MIN).mean(axis=0), (aligned < -STEP_MIN).mean(axis=0))
        j = int(score[EDGE_MARGIN:-EDGE_MARGIN].argmax()) + EDGE_MARGIN
        if best is None or score[j] > best[2]:
            best = (float(x0 + j), float(tilt), float(score[j]))
    return best if best and best[2] >= SEAM_MIN_SCORE else None


def _aligned_strip(gray: np.ndarray, seam: Seam, half: int) -> np.ndarray:
    """(h x 2*half+1): every row cut around the seam line, so the seam is the middle column."""
    h, w = gray.shape
    xs = seam[0] + seam[1] * (np.arange(h) / max(1, h - 1) - 0.5)
    cols = np.clip(np.round(xs)[:, None].astype(int) + np.arange(-half, half + 1)[None, :], 0, w - 1)
    return np.take_along_axis(gray, cols, axis=1)


def _shadow_reach(strip: np.ndarray, ref: float, half: int) -> Tuple[int, int]:
    """Columns left/right of the seam (from the strip's centre) that stay darker than the page's paper."""
    level = np.percentile(strip, 95, axis=0)
    dark = level < ref * (1 - SHADOW_DROP)
    reach = []
    for side in (-1, 1):
        n = 0
        while n < half and dark[half + side * (n + 1)]:
            n += 1
        reach.append(n)
    return reach[0], reach[1]


def gutter_mask(img: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    """(mask 255 over the seam + its shadow/defects, info). Empty mask + found=False without a seam."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)
    seam = find_seam(gray)
    if seam is None:
        return mask, {"found": False}
    half = int(w * SHADOW_MAX_FRAC)
    strip = _aligned_strip(gray, seam, half)
    ref = float(np.percentile(gray[:, int(w * 0.25):int(w * 0.40)], 95) + np.percentile(gray[:, int(w * 0.60):int(w * 0.75)], 95)) / 2
    left, right = _shadow_reach(strip, ref, half)
    pad = max(6, int(w * PAD_FRAC))
    xs = seam[0] + seam[1] * (np.arange(h) / max(1, h - 1) - 0.5)
    for y in range(h):
        mask[y, max(0, int(xs[y]) - left - pad):min(w, int(xs[y]) + right + pad + 1)] = 255
    return mask, {"found": True, "x": round(seam[0]), "tilt": seam[1], "score": round(seam[2], 2), "shadow": (left, right)}


def _overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[mask > 0] = (0.4 * out[mask > 0] + 0.6 * np.array((0, 255, 0))).astype(np.uint8)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for path in a.images:
        img = cv2.imread(path)
        if img is None:
            print(f"SKIP {path}: unreadable", file=sys.stderr); continue
        mask, info = gutter_mask(img)
        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(a.out, f"{stem}_gutter_mask.png"), mask)
        cv2.imwrite(os.path.join(a.out, f"{stem}_gutter_overlay.jpg"), _overlay(img, mask), [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"{stem}: {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
