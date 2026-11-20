#!/usr/bin/env python3
"""Straighten a page scan and find its trim box (for Photoshop guides).

As a module: `deskew(img) -> (straight image, angle)` and `trim_box(img) -> {left, top, right, bottom}` (None = no guide).
As a CLI:     page_guides.py <image> [<image> ...] --out DIR   (writes <stem>_guides.jpg with the box drawn)

Skew: the long straight lines of a page (panel borders, the paper edge, the edge of the art)
are found with a probabilistic Hough transform; the length-weighted median of their
deviation from vertical/horizontal is the scan's rotation (|angle| <= MAX_SKEW deg).

Trim box: on the straightened page, a page edge is a straight step that runs (almost) the
whole height (vertical edges) or width (horizontal ones) - panel borders do not, they stop
at the panel. Per column the share of rows with a strong horizontal step (same sign
consistently) is its coherence; the left edge is the MOST coherent column in the outer
SIDE_ZONE of the width, the right edge likewise; top/bottom the same on rows (between the
left and right edges) within TB_ZONE of the height - deeper rows are panel borders. A side
whose best edge stays under COHERENCE gets no guide (None): place that one by hand.

Example: page_guides.py 080.png --out /tmp/guides
"""
import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

MAX_SKEW = 3.0          # deg; larger is not a scan skew
MIN_LINE_FRAC = 0.25    # Hough lines at least this share of the page's short side
SIDE_ZONE = 0.30        # left/right page edges are looked for within this share of the width from each border
TB_ZONE = 0.10          # top/bottom edges within this share of the height (deeper lines are panel borders)
COHERENCE = 0.60        # share of rows (columns) with a step for a column (row) to be a page edge
STEP = 40               # grey-level step between 3 px on each side


def skew_angle(gray: np.ndarray) -> float:
    """Rotation of the scan in degrees (positive = counter-clockwise content). Example: 0.35."""
    h, w = gray.shape
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, 120, minLineLength=int(MIN_LINE_FRAC * min(h, w)), maxLineGap=6)
    if lines is None:
        return 0.0
    devs, weights = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        dev = ((ang + 45) % 90) - 45            # deviation from the nearest axis
        if abs(dev) <= MAX_SKEW:
            devs.append(dev); weights.append(np.hypot(x2 - x1, y2 - y1))
    if not devs:
        return 0.0
    order = np.argsort(devs)
    cum = np.cumsum(np.array(weights)[order])
    return float(np.array(devs)[order][np.searchsorted(cum, cum[-1] / 2)])


def rotate(img: np.ndarray, angle: float, transparent: bool = True) -> np.ndarray:
    """img rotated by `angle` deg about its centre, same size; emptied corners transparent (BGRA) or replicated."""
    h, w = img.shape[:2]
    if transparent and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if abs(angle) < 0.05:
        return img
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = cv2.BORDER_CONSTANT if transparent else cv2.BORDER_REPLICATE
    return cv2.warpAffine(img, rot, (w, h), flags=cv2.INTER_CUBIC, borderMode=border, borderValue=(0, 0, 0, 0))


def deskew(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """(image rotated straight, angle). Corners left empty are transparent (BGRA out).
    Measure the angle on the scan as it was (before any erasing): erased text changes its long lines."""
    angle = skew_angle(cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY))
    angle = 0.0 if abs(angle) < 0.05 else angle
    return rotate(img, angle), angle


def _coherence(gray: np.ndarray, axis: int) -> np.ndarray:
    """Per column (axis=1) or row (axis=0): share of lines with a strong, same-signed step across it."""
    g = gray.astype(np.int16) if axis == 1 else gray.T.astype(np.int16)
    k = np.ones(3, np.float32) / 3
    sm = cv2.filter2D(g.astype(np.float32), -1, k.reshape(1, 3))
    step = np.zeros_like(sm)
    step[:, 3:-3] = sm[:, 6:] - sm[:, :-6]
    pos, neg = (step > STEP).mean(axis=0), (step < -STEP).mean(axis=0)
    return np.maximum(pos, neg)


def _strongest(coh: np.ndarray, lo: int, hi: int) -> Optional[int]:
    """Index of the most coherent edge in [lo, hi), or None when none reaches COHERENCE."""
    i = lo + int(np.argmax(coh[lo:hi]))
    return i if coh[i] >= COHERENCE else None


def trim_box(img: np.ndarray) -> Dict[str, Optional[int]]:
    """Page edges on a straightened image: {"left","right","top","bottom"} in px, None = not found (no guide)."""
    gray = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    if img.shape[2] == 4:                       # corners emptied by deskew: neutral, so they make no edge
        empty = img[..., 3] == 0
        gray[empty] = int(np.median(gray[~empty]))
    h, w = gray.shape
    cx = _coherence(gray, 1)
    left = _strongest(cx, 3, int(w * SIDE_ZONE))
    right = _strongest(cx, int(w * (1 - SIDE_ZONE)), w - 3)
    x0, x1 = left or 0, right or w
    cy = _coherence(gray[:, x0:x1], 0)
    top = _strongest(cy, 3, int(h * TB_ZONE))
    bottom = _strongest(cy, int(h * (1 - TB_ZONE)), h - 3)
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for path in a.images:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"SKIP {path}: unreadable", file=sys.stderr); continue
        img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        straight, angle = deskew(img)
        box = trim_box(straight)
        view = straight[..., :3].copy()
        h, w = view.shape[:2]
        for side, v in box.items():
            if v is None:
                continue
            p0, p1 = ((v, 0), (v, h)) if side in ("left", "right") else ((0, v), (w, v))
            cv2.line(view, p0, p1, (255, 255, 0), 3)
        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(a.out, f"{stem}_guides.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"{stem}: skew {angle:+.2f} deg, trim box {box}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
