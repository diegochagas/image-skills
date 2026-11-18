#!/usr/bin/env python3
"""Read the EAN-13 / ISBN barcode on a back-cover photo with OpenCV's
BarcodeDetector (no extra dependency). Tries the photo as is, sharpened and
rotated, because phone shots of glossy covers often miss on the first pass.

Example:
    read_barcodes(cv2.imread("back.jpg"))  # [{"type": "EAN_13", "value": "9788545702634", "isbn": "9788545702634"}]
"""
from __future__ import annotations

from typing import TypedDict

import cv2
import numpy as np


class BarcodeHit(TypedDict):
    type: str
    value: str
    isbn: str | None


def isbn_from_ean(value: str) -> str | None:
    """The ISBN-13 when an EAN-13 is a book code (978/979 prefix) with a valid check digit."""
    if len(value) != 13 or not value.isdigit() or value[:3] not in ("978", "979"):
        return None
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(value[:12]))
    return value if (10 - total % 10) % 10 == int(value[12]) else None


def variants(img: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    sharp = cv2.addWeighted(gray, 1.6, cv2.GaussianBlur(gray, (0, 0), 3), -0.6, 0)
    rotations = [cv2.rotate(gray, code) for code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    return [gray, sharp, *rotations]


def read_barcodes(img: np.ndarray) -> list[BarcodeHit]:
    """Every distinct barcode found in the image."""
    detector = cv2.barcode.BarcodeDetector()
    found: dict[str, BarcodeHit] = {}
    for candidate in variants(img):
        ok, values, types, _ = detector.detectAndDecodeWithType(candidate)
        if not ok:
            continue
        for value, kind in zip(values, types):
            if value and value not in found:
                found[value] = {"type": str(kind), "value": value, "isbn": isbn_from_ean(value)}
    return list(found.values())
