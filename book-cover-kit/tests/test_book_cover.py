"""book-cover-kit checks on synthetic images (no photos in the repo).

Run: venv/bin/python -m unittest discover -s book-cover-kit/tests
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cover_barcode import isbn_from_ean  # noqa: E402
from cover_geometry import flatten_cover  # noqa: E402
from cover_palette import cover_palette, readable_text  # noqa: E402


def synthetic_cover() -> np.ndarray:
    """A 300x450 two-colour 'cover' with a bright title band."""
    cover = np.full((450, 300, 3), (40, 30, 160), np.uint8)
    cover[40:120] = (230, 230, 230)
    cv2.circle(cover, (150, 290), 70, (30, 140, 30), -1)
    return cover


def photographed(cover: np.ndarray) -> np.ndarray:
    """The cover warped onto a grey table, seen at an angle."""
    h, w = cover.shape[:2]
    table = np.full((900, 800, 3), (95, 105, 110), np.uint8)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[180, 130], [560, 170], [600, 760], [150, 720]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(cover, matrix, (800, 900))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, (800, 900))
    table[mask > 0] = warped[mask > 0]
    return table


class BookCoverKitTest(unittest.TestCase):
    def test_flatten_recovers_the_cover_aspect(self) -> None:
        flat, quad = flatten_cover(photographed(synthetic_cover()))
        self.assertIsNotNone(quad)
        self.assertAlmostEqual(flat.shape[1] / flat.shape[0], 300 / 450, delta=0.05)

    def test_flat_scan_is_only_trimmed(self) -> None:
        bordered = cv2.copyMakeBorder(synthetic_cover(), 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        flat, quad = flatten_cover(bordered)
        self.assertEqual(flat.shape[:2], (450, 300))
        self.assertIsNone(quad)

    def test_palette_picks_the_dominant_colour_and_readable_text(self) -> None:
        palette = cover_palette(synthetic_cover())
        self.assertEqual(palette["spine"], "#a01e28")
        self.assertEqual(palette["text"], "#f7f3ea")
        self.assertEqual(readable_text(np.array([240.0, 240.0, 240.0])), "#111111")

    def test_isbn_needs_a_book_prefix_and_valid_check_digit(self) -> None:
        self.assertEqual(isbn_from_ean("9788545702634"), "9788545702634")
        self.assertIsNone(isbn_from_ean("9788545702635"))
        self.assertIsNone(isbn_from_ean("7891234567895"))


class ShrinkCommandTest(unittest.TestCase):
    def test_shrink_resizes_without_touching_geometry(self) -> None:
        import tempfile

        from book_cover import main

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in"
            src.mkdir()
            cv2.imwrite(str(src / "a.png"), cv2.resize(synthetic_cover(), (1200, 1800)))
            main(["shrink", str(src), "--output", str(Path(tmp) / "out"), "--max-side", "300"])
            out = cv2.imread(str(Path(tmp) / "out" / "a.jpg"))
            self.assertEqual(out.shape[:2], (300, 200))


if __name__ == "__main__":
    unittest.main()
