"""The four corrections gutter_local.py applies to the strip around a spread's seam (steps 3-6 of its
docstring). Each takes a gutter_local.Strip and changes strip.data in place; distances are units
(strip.D) of the 1500 px reference spread, row windows are scaled by strip.s.

Example: veil(strip); dividing_line(strip); dead_core(strip); even_tones(strip)
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, percentile_filter

from gutter_local import DEAD_GAIN, fill_gaps, per_row, row_bins

BLACK_MAX, PAPER_MIN = 110, 215      # reference columns this dark / this light count as black ground / paper


def _cols(strip, d0: float, d1: float) -> np.ndarray:
    return np.nonzero((strip.D >= d0) & (strip.D <= d1))[0]


def veil(strip) -> None:
    """Undo the flare (black lifted towards white) and the shadow (white darkened) next to the spine."""
    g, D, s, h = strip.gray(), strip.D, strip.s, strip.h
    Y = row_bins(h, s)
    F = np.full((len(Y), len(D)), np.nan)
    S = np.full_like(F, np.nan)
    win = int(24 * s)
    refs = {-1: _cols(strip, -48, -28), 1: _cols(strip, 44, 62)}
    for i, y in enumerate(Y):
        blk = g[max(0, y - win):min(h, y + win)]
        if len(blk) < 16:
            continue
        b, w = np.percentile(blk, 8, axis=0), np.percentile(blk, 95, axis=0)
        for side, rc in refs.items():
            ref = blk[:, rc]
            rb, rw = np.percentile(ref, 8), np.percentile(ref, 95)
            sel = (np.sign(D) == side) & (np.abs(D) > 1) & (np.abs(D) < 50)
            if rb < BLACK_MAX:
                F[i, sel & (b < 200)] = np.maximum(0, (b[sel & (b < 200)] - rb) / (255 - rb))
            if rw > PAPER_MIN:
                S[i, sel & (w > 170)] = np.minimum(1, w[sel & (w > 170)] / rw)
    F = gaussian_filter1d(fill_gaps(F, Y, 0, 400 * s), 2.0, axis=0)
    S = gaussian_filter1d(fill_gaps(S, Y, 1, 40 * s), 2.0, axis=0)
    F = gaussian_filter1d(np.where(F < 0.03, 0, F), 1.0 * s, axis=1)     # noise in the percentiles is not a veil
    S = gaussian_filter1d(np.where(S > 0.96, 1, S), 1.0 * s, axis=1)
    for side in (-1, 1):                     # the columns right at the line take the lift of |d| = 7
        j7 = strip.j(7 * side)
        near = (np.sign(D) == side) & (np.abs(D) >= 2) & (np.abs(D) < 7)
        F[:, near] = np.maximum(F[:, near], F[:, [j7]])
    S[:, np.abs(D) < 8] = S[:, [strip.j(-8)]] * (D[np.abs(D) < 8] < 0) + S[:, [strip.j(8)]] * (D[np.abs(D) < 8] >= 0)
    ramp = np.clip((50 - np.abs(D)) / 6, 0, 1)                 # the line's own shading is dividing_line's job
    F = np.clip(per_row(F * ramp, Y, h), 0, 0.8)
    S = np.clip(per_row(1 - (1 - S) * ramp, Y, h), 0.3, 1)
    strip.data = np.clip((strip.data - 255 * F[..., None]) / (1 - F[..., None]) / S[..., None], 0, 255)


def dividing_line(strip) -> np.ndarray:
    """Undo each seam column's gain/offset (paper and black anchors); returns the per-pixel gain map."""
    g, D, s, h = strip.gray(), strip.D, strip.s, strip.h
    Y = row_bins(h, s)
    jl, jr = strip.j(-8), strip.j(10)
    cols = _cols(strip, -5, 7)
    OP, RP, OB, RB = [np.full((len(Y), len(cols)), np.nan) for _ in range(4)]
    win = int(20 * s)
    t = (D[cols] - D[jl]) / (D[jr] - D[jl])
    for i, y in enumerate(Y):
        rows = slice(max(0, y - win), min(h, y + win + 1))
        l, r = g[rows, jl], g[rows, jr]
        ref = l[:, None] * (1 - t) + r[:, None] * t
        obs = g[rows][:, cols]
        paper, black = (l > 200) & (r > 200), (l < 100) & (r < 100)
        if paper.sum() >= 6:
            OP[i], RP[i] = np.median(obs[paper], 0), np.median(ref[paper], 0)
        if black.sum() >= 6:
            OB[i], RB[i] = np.median(obs[black], 0), np.median(ref[black], 0)
    # both anchors are needed; where a band has none nearby (screentone, mid greys) the column is left alone
    OP, RP, OB, RB = (fill_gaps(M, Y, np.nan, 150 * s) for M in (OP, RP, OB, RB))
    A = np.clip((OP - OB) / np.maximum(RP - RB, 20), 0.2, 1.6)
    B = OP - A * RP
    A, B = fill_gaps(A, Y, 1.0, 150 * s), fill_gaps(B, Y, 0.0, 150 * s)
    A, B = per_row(gaussian_filter1d(A, 1.5, axis=0), Y, h), per_row(gaussian_filter1d(B, 1.5, axis=0), Y, h)
    seg = strip.data[:, cols]
    rec = (seg - B[..., None]) / A[..., None]
    lv, rv = strip.data[:, jl].mean(1), strip.data[:, jr].mean(1)
    mid = np.maximum(lv, rv) < 180                       # two mid tones: nothing much darker than the darker side
    lo, hi = (np.minimum(lv, rv) - 5)[:, None, None], (np.maximum(lv, rv) + 10)[:, None, None]
    rec = np.where(mid[:, None, None], np.clip(rec, lo, hi), rec)
    w = np.clip((A - DEAD_GAIN) / 0.1, 0, 1)[..., None]
    strip.data[:, cols] = np.clip(w * rec + (1 - w) * seg, 0, 255)
    gain = np.ones((h, len(D)))
    gain[:, cols] = A
    return gain


def _direction(strip, g: np.ndarray, y: int, a: int, b: int, J) -> tuple:
    """(slope in rows per column along which the core of row y is painted, colour mismatch along it)."""
    slopes = np.tan(np.radians(np.arange(-70, 71, 5)))
    J11, J22, J12 = J
    acc = np.zeros(3)
    for x in (a - int(2 * strip.s), b + int(2 * strip.s)):
        x = int(np.clip(x, 0, g.shape[1] - 1))
        tr = J11[y, x] + J22[y, x]
        coh = np.sqrt((J11[y, x] - J22[y, x]) ** 2 + 4 * J12[y, x] ** 2) / (tr + 1e-3)
        if tr >= 6000 and coh >= 0.6:
            acc += (J11[y, x], J22[y, x], J12[y, x])
    if acc.any():
        lt = 0.5 * np.arctan2(2 * acc[2], acc[0] - acc[1]) + np.pi / 2
        if abs(np.cos(lt)) >= 0.34:
            return float(np.sin(lt) / np.cos(lt)), 0.0
    ar = np.arange(g.shape[0])
    cost = [abs(np.interp(y + sl * (a - 1 - (a + b) / 2), ar, g[:, a - 1]) - np.interp(y + sl * (b + 1 - (a + b) / 2), ar, g[:, b + 1]))
            + abs(np.degrees(np.arctan(sl))) * 0.4 for sl in slopes]
    k = int(np.argmin(cost))
    return float(slopes[k]), float(cost[k] - abs(np.degrees(np.arctan(slopes[k]))) * 0.4)


def line_columns(g: np.ndarray, D: np.ndarray, k: int = 4) -> np.ndarray:
    """Columns of the dark dividing line itself: the run, around the darkest column near the seam, whose
    median over the height is >= 15 levels darker than the columns k px either side (a few px wide)."""
    dark = np.zeros(g.shape[1])
    dark[k:-k] = np.median(g[:, k:-k] - (g[:, :-2 * k] + g[:, 2 * k:]) / 2, axis=0)
    near = np.nonzero(np.abs(D) <= 5)[0]
    c = int(near[np.argmin(dark[near])])
    core = np.zeros(g.shape[1], bool)
    if dark[c] > -15:
        return core
    a = b = c
    while a - 1 >= near[0] and dark[a - 1] < -8:
        a -= 1
    while b + 1 <= near[-1] and dark[b + 1] < -8:
        b += 1
    core[a:b + 1] = True
    return core


def dead_core(strip, gain: np.ndarray, rng=np.random.default_rng(7)) -> None:
    """Paint the columns the line left without image: blend of both sides along the drawing's direction."""
    g = strip.gray()
    gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    J = tuple(cv2.GaussianBlur(m, (0, 0), 3) for m in (gx * gx, gy * gy, gx * gy))
    ar = np.arange(strip.h)
    core = line_columns(g, strip.D)
    near_core = np.convolve(core, np.ones(5), "same") > 0          # a low gain counts only next to the line
    for y in range(strip.h):
        dead = np.nonzero(((gain[y] < DEAD_GAIN + 0.05) & near_core) | core)[0]
        if not len(dead):
            continue
        a, b = int(dead[0]), int(dead[-1])
        sl, mismatch = _direction(strip, g, y, a, b, J)
        ks = np.arange(a, b + 1)
        t = ((ks - (a - 1)) / (b - a + 2))[:, None]
        cl = np.stack([np.interp(y + sl * (a - 1 - ks), ar, strip.data[:, a - 1, c]) for c in range(3)], 1)
        cr = np.stack([np.interp(y + sl * (b + 1 - ks), ar, strip.data[:, b + 1, c]) for c in range(3)], 1)
        fill = cl * (1 - t) + cr * t
        c0, c1 = strip.data[y, a - 1], strip.data[y, b + 1]
        if mismatch > 60 and min(c0.mean(), c1.mean()) < 110:                    # ink meets paper: ink wins
            fill = np.repeat((c0 if c0.mean() < c1.mean() else c1)[None], len(ks), 0)
        if max(c0.mean(), c1.mean()) < 110:                                      # black ground: its grain
            nb = np.concatenate([g[max(0, y - 2):y + 3, max(0, a - 7):a].ravel(), g[max(0, y - 2):y + 3, b + 1:b + 8].ravel()])
            fill += rng.normal(0, float(np.clip(nb[nb < np.percentile(nb, 80)].std(), 0, 6)), (len(ks), 1))
        strip.data[y, a:b + 1] = np.clip(fill, 0, 255)


def even_tones(strip) -> None:
    """Black level (black ground) and paper white across the band on a straight ramp between the pages."""
    D, s, h = strip.D, strip.s, strip.h
    step = max(1, int(round(s)))
    g = strip.gray()[::step]
    Lb = gaussian_filter(percentile_filter(g, 12, size=(41, 3)), (6, 1.2))
    jl, jr = strip.j(-40), strip.j(40)
    t = np.clip((D + 40) / 80, 0, 1)
    delta = Lb[:, [jl]] * (1 - t) + Lb[:, [jr]] * t - Lb
    delta[(Lb > 200) | (Lb[:, [jl]] > 120) | (Lb[:, [jr]] > 120)] = 0     # a lifted black (veil) is fixed too
    delta = gaussian_filter(delta, (4, 1.0)) * np.clip((44 - np.abs(D)) / 4, 0, 1)
    up = lambda m: cv2.resize(m.astype(np.float32), (m.shape[1], h), interpolation=cv2.INTER_LINEAR)
    delta, Lb = up(delta), up(Lb)
    v = strip.data
    strip.data = np.clip(v + (delta * (255 - v.mean(2)) / np.maximum(255 - Lb, 1))[..., None], 0, 255)
    g = strip.gray()[::step]
    Lw = gaussian_filter(percentile_filter(g, 85, size=(15, 1)), (3, 0.6))
    kl, kr = strip.j(-14), strip.j(14)
    t2 = np.clip((D + 14) / 28, 0, 1)
    ok = (Lw[:, [kl]] > 200) & (Lw[:, [kr]] > 200) & (Lw > 150)
    gain = np.where(ok, (Lw[:, [kl]] * (1 - t2) + Lw[:, [kr]] * t2) / np.maximum(Lw, 1), 1.0)
    gain = np.clip(gaussian_filter(gain, (2, 0.5)), 0.9, 1.25)
    gain[:, np.abs(D) > 14] = 1
    gain = up(gain)
    v = strip.data
    light = np.clip((v.mean(2) - 120) / 80, 0, 1)                                 # ink stays
    strip.data = np.clip(v * (1 + light * (gain - 1))[..., None], 0, 255)
