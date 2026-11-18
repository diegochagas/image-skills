#!/usr/bin/env python3
"""Find a book cover in a phone photo and warp it flat.

A cover photographed on a table is a bright-or-dark quadrilateral on a
background, seen at an angle. find_cover_quad() looks for the largest convex
four-corner contour (several edge/threshold passes, the one covering the most
of the photo wins); flatten_cover() warps it to an upright rectangle whose
aspect comes from the quad's own edge lengths. When no quad is found the
photo is only trimmed of a uniform border, so an already-flat scan passes
through unchanged.

Example:
    img = cv2.imread("photo.jpg")
    flat, quad = flatten_cover(img)
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_AREA_FRAC = 0.12   # a cover smaller than this share of the photo is not the subject
MAX_AREA_FRAC = 0.97   # a quad this big is the photo frame itself: the cover is already flat
WORK_SIDE = 900        # contours are searched on a copy this long on its longest side
APPROX_EPS = 0.02      # polygon approximation tolerance (fraction of the perimeter)
AXIS_TOLERANCE_DEG = 1.5  # a quad whose edges are all this close to the axes is already flat
TRIM_TOLERANCE = 18    # max colour distance from the corner colour still counted as border


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = points.reshape(4, 2).astype(np.float32)
    sums, diffs = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(sums)], pts[np.argmin(diffs)], pts[np.argmax(sums)], pts[np.argmax(diffs)]], dtype=np.float32)


def candidate_masks(gray: np.ndarray) -> list[np.ndarray]:
    """Binary images whose outer contours may outline the cover."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(blur, 40, 140), np.ones((5, 5), np.uint8), iterations=2)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((9, 9), np.uint8)
    return [edges, cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel), cv2.morphologyEx(255 - otsu, cv2.MORPH_CLOSE, kernel)]


def largest_quad(mask: np.ndarray, min_area: float) -> np.ndarray | None:
    """Biggest convex 4-corner polygon among the mask's outer contours, the photo frame excluded."""
    max_area = MAX_AREA_FRAC * mask.shape[0] * mask.shape[1]
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, min_area
    for contour in contours:  # RETR_LIST below would also see a cover inside a frame-sized blob
        hull = cv2.convexHull(contour)
        poly = cv2.approxPolyDP(hull, APPROX_EPS * cv2.arcLength(hull, True), True)
        area = cv2.contourArea(poly)
        if len(poly) == 4 and best_area < area < max_area:
            best, best_area = poly, area
    return best


def find_cover_quad(img: np.ndarray) -> np.ndarray | None:
    """Corners (TL, TR, BR, BL) of the cover in full-size pixel coordinates, or None."""
    h, w = img.shape[:2]
    scale = WORK_SIDE / max(h, w)
    small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    min_area = MIN_AREA_FRAC * small.shape[0] * small.shape[1]
    quads = [q for q in (largest_quad(m, min_area) for m in candidate_masks(gray)) if q is not None]
    if not quads:
        return None
    return order_corners(max(quads, key=cv2.contourArea) / scale)


def warp_quad(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-warp the quad to an upright rectangle sized by its mean edge lengths."""
    tl, tr, br, bl = quad
    width = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
    height = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), target)
    return cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_CUBIC)


def trim_border(img: np.ndarray) -> np.ndarray:
    """Cut a uniform border (the colour of the top-left corner) off all four sides."""
    corner = img[0, 0].astype(np.int16)
    distance = np.abs(img.astype(np.int16) - corner).max(axis=2)
    rows, cols = np.where(distance > TRIM_TOLERANCE)
    if rows.size == 0:
        return img
    return img[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]


def is_axis_aligned(quad: np.ndarray, tolerance_deg: float = AXIS_TOLERANCE_DEG) -> bool:
    """True when every edge of the quad runs within tolerance of horizontal/vertical."""
    for start, end in zip(quad, np.roll(quad, -1, axis=0)):
        dx, dy = end - start
        angle = abs(np.degrees(np.arctan2(dy, dx))) % 90
        if min(angle, 90 - angle) > tolerance_deg:
            return False
    return True


def flatten_cover(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """(flat cover image, quad used or None when the photo was only trimmed)."""
    quad = find_cover_quad(img)
    # An upright quad means a scan (or a straight shot): nothing to unwarp, and on a
    # light border the quad may have snapped to part of the cover (a white title band
    # merges with white paper), so trimming the border is the safer cut.
    if quad is None or is_axis_aligned(quad):
        return trim_border(img), None
    return warp_quad(img, quad), quad
