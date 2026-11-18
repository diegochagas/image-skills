#!/usr/bin/env python3
"""Colours a book cover gives a 3D shelf: the spine colour (dominant cover
colour, k-means on a downscaled copy), a readable text colour on it (black or
white by WCAG contrast) and an accent (the most saturated large cluster).

Example:
    palette = cover_palette(cv2.imread("cover.jpg"))
    palette["spine"]  # "#7a1f2b"
"""
from __future__ import annotations

from typing import TypedDict

import cv2
import numpy as np

CLUSTERS = 5
SAMPLE_SIDE = 96         # k-means runs on a copy this long on its longest side
MIN_ACCENT_SHARE = 0.08  # an accent cluster must cover at least this share of the cover


class CoverPalette(TypedDict):
    spine: str
    text: str
    accent: str
    shares: list[float]


def to_hex(bgr: np.ndarray) -> str:
    b, g, r = (int(round(c)) for c in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def relative_luminance(bgr: np.ndarray) -> float:
    """WCAG relative luminance of a BGR colour (0..1)."""
    channels = [c / 255 for c in bgr[::-1]]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def readable_text(bgr: np.ndarray) -> str:
    """'#111111' or '#f7f3ea', whichever contrasts more with the colour."""
    lum = relative_luminance(bgr)
    on_dark = (1.05) / (lum + 0.05)
    on_light = (lum + 0.05) / (0.0067 + 0.05)
    return "#f7f3ea" if on_dark >= on_light else "#111111"


def cluster_colours(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cluster centres BGR, share of pixels per centre), largest first."""
    h, w = img.shape[:2]
    scale = SAMPLE_SIDE / max(h, w)
    small = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)
    k = min(CLUSTERS, len(pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(7)  # deterministic output for the same cover
    _, labels, centres = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=k) / len(labels)
    order = np.argsort(-shares)
    return centres[order], shares[order]


def saturation(bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0, 1])


def cover_palette(img: np.ndarray) -> CoverPalette:
    """Spine, text and accent colours of a cover image (BGR array)."""
    centres, shares = cluster_colours(img)
    spine = centres[0]
    large = [c for c, s in zip(centres, shares) if s >= MIN_ACCENT_SHARE]
    accent = max(large, key=saturation) if large else spine
    return {
        "spine": to_hex(spine),
        "text": readable_text(spine),
        "accent": to_hex(accent),
        "shares": [round(float(s), 3) for s in shares],
    }
