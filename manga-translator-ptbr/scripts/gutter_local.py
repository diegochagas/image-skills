#!/usr/bin/env python3
"""Repair the middle of a double-page scan locally (free, no AI): align the halves, undo the bright
veil / shadow next to the spine, remove the dividing line, paint its dead core, even the tones.

As a module: `repair_gutter(page_bgr, seam) -> (image, info)`  (seam = gutter_mask's (x_mid, tilt, score))
As a CLI:    gutter_local.py <spread.png> --out DIR   (writes <stem>_gutter.png and a before/after strip)

Worked out by hand on a 1500 px wide Saint Seiya spread (2026-11-20); every distance below is in
"units" of that spread and scaled by s = width / 1500. In the strip around the seam, d = x - seam(y):
  1. seam: the dark dividing line is fitted (robust line through the per-band darkest thin ridge
     near gutter_mask's seam); without a clear line gutter_mask's seam is used.
  2. alignment: straight lines that cross the band are found on both sides (LSD), extended to the
     seam and paired by angle; the median height difference (>= 3 pairs, |shift| <= MAX_SHIFT units)
     moves the right half. No horizontal space is added: the measured gap was 0.
  3. veil / shadow: per column, its black level vs untouched columns of the same page gives the
     flare (obs = t*(1-f) + 255*f), its white level the shadow (obs = t*s); both are divided out.
  4. dividing line: per column, gain and offset from paper (both sides white) and black ground
     (both sides dark) rows are undone, so thin art inside the line survives; where the gain is
     under DEAD_GAIN nothing is left of the image and the core is painted (5).
  5. dead core: each pixel is a blend of the colours just outside the core met along the drawing's
     direction (structure tensor), else the direction where both sides match best; a dark line
     meeting paper keeps the ink; grain like the neighbours on black ground.
  6. tones: the black level (and the paper white) across the band is put on a straight ramp
     between the two pages, so no strip stands out.
Lines that cross the band blurred are NOT redrawn (that was done by hand with the brush).

Example: gutter_local.py ~/Downloads/x/084-085.png --out /tmp/gl
"""
import argparse
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, percentile_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gutter_mask  # noqa: E402

REF_W = 1500.0          # width of the spread the units come from
MAX_SHIFT = 6           # units
DEAD_GAIN = 0.38        # under this column gain the line left no image
REACH = (-48, 62)       # units of the strip handled around the seam
Seam = Tuple[float, float, float]
Line = Callable[[np.ndarray], np.ndarray]


def sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear samples of img at float coordinates (same shape as xs)."""
    return cv2.remap(img, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def fit_seam(gray: np.ndarray, seam: Seam, s: float) -> Line:
    """x(y) of the dark dividing line: robust fit of the per-band thin dark ridge, else gutter_mask's line."""
    h = gray.shape[0]
    base = lambda y: seam[0] + seam[1] * (y / max(1, h - 1) - 0.5)
    k = cv2.GaussianBlur(gray.astype(np.float32), (1, 0), sigmaX=0.1, sigmaY=8 * s)
    off = max(2, int(round(4 * s)))
    resp = np.zeros_like(k)
    resp[:, off:-off] = (k[:, :-2 * off] + k[:, 2 * off:]) / 2 - k[:, off:-off]
    ys, xs = [], []
    reach = int(40 * s)
    for y in range(int(0.04 * h), int(0.96 * h), max(4, int(8 * s))):
        x0 = int(base(y))
        win = resp[y, x0 - reach:x0 + reach]
        if win.size and win.max() > 10:
            ys.append(y); xs.append(x0 - reach + int(win.argmax()))
    if len(ys) < 20:
        return base
    ys, xs = np.array(ys, float), np.array(xs, float)
    keep = np.ones(len(ys), bool)
    for _ in range(4):
        p = np.polyfit(ys[keep], xs[keep], 1)
        keep = np.abs(xs - np.polyval(p, ys)) < 2.5 * s
    if keep.mean() < 0.5:
        return base
    return lambda y: p[0] * y + p[1]


def _segments(gray: np.ndarray, line: Line, s: float, side: int) -> List[Tuple[float, float, float]]:
    """Straight segments ending near the seam on one side: (slope, y where they meet the seam, length)."""
    lsd = cv2.createLineSegmentDetector(0)
    found = lsd.detect(gray)[0]
    out = []
    for x1, y1, x2, y2 in (found.reshape(-1, 4) if found is not None else []):
        if abs(x2 - x1) < 1:
            continue
        k = (y2 - y1) / (x2 - x1)
        length = float(np.hypot(x2 - x1, y2 - y1))
        if abs(k) > 1.5 or length < 20 * s:
            continue
        d1, d2 = x1 - line(np.array(y1)), x2 - line(np.array(y2))
        near = min(abs(d1), abs(d2))
        if np.sign(d1) != side or np.sign(d2) != side or near > 45 * s or near < 4 * s:
            continue
        xi, yi = (x1, y1) if abs(d1) < abs(d2) else (x2, y2)
        for _ in range(3):                                  # where the segment's line crosses the seam
            yi = yi + k * (float(line(np.array(yi))) - xi); xi = float(line(np.array(yi)))
        out.append((float(k), float(yi), length))
    return out


def vertical_shift(gray: np.ndarray, line: Line, s: float) -> int:
    """Rows the right half sits lower than the left (positive = move it up); 0 when unclear.

    Lines that cross the band are found on both sides (LSD), extended to the seam and paired by
    angle; the median height difference of >= 3 pairs is the shift. Example: 3 on the test spread."""
    left, right = _segments(gray, line, s, -1), _segments(gray, line, s, 1)
    dys = []
    for kl, yl, _ in left:
        best = None
        for kr, yr, _ in right:
            if abs(np.degrees(np.arctan(kl) - np.arctan(kr))) < 2.5 and abs(yr - yl) <= MAX_SHIFT * s * 1.5:
                best = yr - yl if best is None or abs(yr - yl) < abs(best) else best
        if best is not None:
            dys.append(best)
    if len(dys) < 3:
        return 0
    dy = float(np.median(dys))
    return int(round(dy)) if abs(dy) <= MAX_SHIFT * s else 0


def shift_right(img: np.ndarray, line: Line, dy: int) -> np.ndarray:
    """img with the half right of the seam moved up by dy rows (rows it leaves keep the original)."""
    if dy == 0:
        return img
    h, w = img.shape[:2]
    moved = np.roll(img, -dy, axis=0)
    if dy > 0:
        moved[h - dy:] = img[h - dy:]
    else:
        moved[:-dy] = img[:-dy]
    right = np.arange(w)[None, :] >= line(np.arange(h))[:, None]
    return np.where(right[..., None], moved, img)


class Strip:
    """The band around the seam resampled so column j is d = D[j] units from the line, every row."""

    def __init__(self, img: np.ndarray, line: Line, s: float):
        self.h = img.shape[0]
        self.s, self.line = s, line
        self.D = np.arange(REACH[0] * s, REACH[1] * s + 1) / s          # one column per pixel, in units
        y = np.arange(self.h, dtype=np.float32)[:, None]
        self.xs = line(y) + self.D[None, :] * s
        self.ys = np.repeat(y, len(self.D), axis=1)
        self.img = img
        self.data = sample(img.astype(np.float32), self.xs, self.ys)

    def gray(self) -> np.ndarray:
        return self.data.mean(axis=2)

    def j(self, d: float) -> int:
        return int(np.argmin(np.abs(self.D - d)))

    def write_back(self, before: np.ndarray) -> np.ndarray:
        """Original image + the change made in the strip, mapped back per row: only the change is resampled,
        except on the line's own columns, whose new pixels are taken as they are (sharp art there was a
        dark line, adding a resampled change to it leaves a ghost)."""
        delta = self.data - before
        out = self.img.astype(np.float32)
        x0 = int(np.floor(self.xs.min())) + 1
        x1 = int(np.ceil(self.xs.max())) - 1
        xx = np.arange(max(0, x0), min(out.shape[1], x1))
        for y in range(self.h):
            d = (xx - self.line(y)) / self.s
            core = np.abs(d - 1) <= 8          # the line and its painted core: the new pixels themselves
            for c in range(3):
                out[y, xx, c] += np.interp(d, self.D, delta[y, :, c], left=0, right=0)
                out[y, xx[core], c] = np.interp(d[core], self.D, self.data[y, :, c])
        return np.clip(out, 0, 255).astype(np.uint8)


def row_bins(h: int, s: float) -> np.ndarray:
    return np.arange(0, h, max(2, int(6 * s)))


def fill_gaps(M: np.ndarray, Y: np.ndarray, default: float, maxgap: float) -> np.ndarray:
    """NaN gaps of each column bridged by interpolation up to maxgap rows away, `default` beyond (may be NaN)."""
    for j in range(M.shape[1]):
        c = M[:, j]
        ok = ~np.isnan(c)
        if ok.sum() < 3:
            M[:, j] = default
            continue
        idx = np.nonzero(ok)[0]
        dist = np.min(np.abs(np.arange(len(Y))[:, None] - idx[None, :]), axis=1) * (Y[1] - Y[0])
        M[:, j] = np.where(dist <= maxgap, np.interp(Y, Y[ok], c[ok]), default)
    return M


def per_row(M: np.ndarray, Y: np.ndarray, h: int) -> np.ndarray:
    """(len(Y) x n) map -> (h x n) by interpolating along y."""
    return np.stack([np.interp(np.arange(h), Y, M[:, j]) for j in range(M.shape[1])], axis=1)


def repair_gutter(page: np.ndarray, seam: Seam) -> Tuple[np.ndarray, Dict[str, object]]:
    """(page with the middle repaired, info). Example: info = {"shift": 3, "line": "fitted"}."""
    import gutter_strip_fixes as fx
    s = page.shape[1] / REF_W
    gray = cv2.cvtColor(page[..., :3], cv2.COLOR_BGR2GRAY)
    line = fit_seam(gray, seam, s)
    dy = vertical_shift(gray, line, s)
    img = shift_right(page[..., :3], line, dy)
    strip = Strip(img, line, s)
    before = strip.data.copy()
    fx.veil(strip)
    gain = fx.dividing_line(strip)
    fx.dead_core(strip, gain)
    fx.even_tones(strip)
    return strip.write_back(before), {"shift": dy, "x_mid": round(float(line(np.array(page.shape[0] / 2))), 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    page = cv2.imread(a.image)
    if page is None:
        sys.exit(f"cannot read {a.image}")
    _, info = gutter_mask.gutter_mask(page)
    if not info["found"]:
        sys.exit(f"no seam found in {a.image}")
    out, rep = repair_gutter(page, (float(info["x"]), float(info["tilt"]), float(info["score"])))
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.image))[0]
    cv2.imwrite(os.path.join(a.out, f"{stem}_gutter.png"), out)
    x = int(rep["x_mid"]); r = int(60 * page.shape[1] / REF_W)
    cv2.imwrite(os.path.join(a.out, f"{stem}_before_after.jpg"),
                np.hstack([page[:, x - r:x + r], np.full((page.shape[0], 6, 3), 255, np.uint8), out[:, x - r:x + r]]))
    print(f"{stem}: seam {info}, {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
