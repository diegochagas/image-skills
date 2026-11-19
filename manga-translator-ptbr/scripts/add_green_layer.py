#!/usr/bin/env python3
"""Add a chroma-green "Green screen" layer to finished PSDs: text over art + the middle of double pages.

    add_green_layer.py <psd> [<psd> ...] [--output-dir DIR] [--name NAME] [--grow 3] [--no-gutter]

Per PSD (its bottom layer, "Original", is the page):
  1. detect_blocks.py + clean_blocks.py run on the Original; the strokes that
     clean_blocks leaves untouched because the background is not one colour
     (text over a character, screentone, speed lines...) are the "over art" mask.
     Grown by --grow px so the lines around the letters are covered too: the
     layer has the exact size of the text, outline included.
  2. A double page (name NNN-MMM, or wider than 1.15 x its height) also gets
     the middle line, the shadow next to it and the scan defects around it
     (gutter_mask.py).
  3. The mask is painted CHROMA_GREEN on a transparent layer, placed over the
     Original's pixels and under the text boxes, above every other raster layer.
Text on a single colour is not marked (it is already erased on Copy).

Output: <DIR>/<psd name> (default ~/Downloads/<source folder> green, or
$COMIC_OUTPUT_DIR/...); the input PSDs are never modified.

Example: add_green_layer.py ~/Downloads/Portuguese/069.psd ~/Downloads/Portuguese/072-073.psd
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gutter_mask  # noqa: E402

PY = sys.executable
CHROMA_GREEN = (0, 177, 64)    # the green of green-screen sets, RGB
DOUBLE_RATIO = 1.15
BLOCK_MARGIN = 6      # px around a text block still counted as "inside"
ART_SHARE = 0.5       # a block keeps its art strokes when they are at least this share of its erased text


def run(cmd: List[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])}...: {res.stderr.strip()[-400:]}")
    return res.stdout


def layers_of(psd: str) -> List[Dict[str, object]]:
    return json.loads(run(["node", os.path.join(HERE, "list_layers.mjs"), psd]))


def layer_below_text(layers: List[Dict[str, object]]) -> str:
    """Name of the last layer under the first text box (the new layer goes right above it).

    Example: Original, Copy, Solid, Text 1, Levels 1 -> "Solid" (adjustment layers above the text stay on top)."""
    first_text = next((l["index"] for l in layers if l["isText"]), len(layers))
    return next(l["name"] for l in reversed(layers) if l["index"] < first_text)


def is_double(stem: str, width: int, height: int) -> bool:
    """A spread scan: named NNN-MMM or clearly landscape. Example: is_double("072-073", 3632, 2586) -> True."""
    return bool(re.fullmatch(r"\d+-\d+", stem)) or width > DOUBLE_RATIO * height


def art_text_mask(page_png: str, work: str, stem: str) -> np.ndarray:
    """Strokes of text over art (255), from detect_blocks + clean_blocks on the page image."""
    run([PY, os.path.join(HERE, "detect_blocks.py"), work, page_png])
    det = os.path.join(work, "detect", f"{stem}_detect.json")
    run([PY, os.path.join(HERE, "clean_blocks.py"), det])
    mask = cv2.imread(os.path.join(work, "detect", f"{stem}_art_text_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"clean_blocks wrote no art mask for {stem}")
    fill = cv2.imread(os.path.join(work, "detect", f"{stem}_fill_text_mask.png"), cv2.IMREAD_GRAYSCALE)
    return keep_art_blocks(mask, fill, json.load(open(det))["text_blocks"])


def keep_art_blocks(art: np.ndarray, fill: np.ndarray, blocks: List[List[int]]) -> np.ndarray:
    """Keep the art strokes of the blocks whose text is mostly over art.

    A block whose letters were erased with a solid fill (white on a black burst) leaves only
    art edges behind (the burst's outline): those are not text. Example: 50 px art vs 900 px
    filled in a block -> that block's art is dropped; 800 vs 100 -> kept."""
    kept = np.zeros_like(art)
    for x, y, w, h in blocks:
        y0, x0 = max(0, y - BLOCK_MARGIN), max(0, x - BLOCK_MARGIN)
        a = art[y0:y + h + BLOCK_MARGIN, x0:x + w + BLOCK_MARGIN]
        f = fill[y0:y + h + BLOCK_MARGIN, x0:x + w + BLOCK_MARGIN]
        n_art, n_fill = int((a > 0).sum()), int((f > 0).sum())
        if n_art and n_art >= ART_SHARE * (n_art + n_fill):
            kept[y0:y + h + BLOCK_MARGIN, x0:x + w + BLOCK_MARGIN] |= a
    return kept


def green_png(mask: np.ndarray, path: str) -> None:
    """RGBA PNG: CHROMA_GREEN where mask, transparent elsewhere."""
    rgba = np.zeros(mask.shape + (4,), np.uint8)
    rgba[mask > 0] = (CHROMA_GREEN[2], CHROMA_GREEN[1], CHROMA_GREEN[0], 255)   # BGRA for cv2
    cv2.imwrite(path, rgba)


def process(psd: str, out_dir: str, a: argparse.Namespace) -> str:
    stem = os.path.splitext(os.path.basename(psd))[0]
    layers = layers_of(psd)
    orig = layers[0]
    with tempfile.TemporaryDirectory(prefix="green_") as work:
        page = os.path.join(work, f"{stem}.png")
        run(["node", "--max-old-space-size=4096", os.path.join(HERE, "export_layer.mjs"), psd, "0", page])
        mask = art_text_mask(page, work, stem)
        n_text = int((mask > 0).any())
        mask = cv2.dilate(mask, np.ones((2 * a.grow + 1,) * 2, np.uint8)) if a.grow else mask
        gutter = None
        if not a.no_gutter and is_double(stem, orig["width"], orig["height"]):
            gm, gutter = gutter_mask.gutter_mask(cv2.imread(page))
            mask = np.maximum(mask, gm)
        layer_png = os.path.join(work, "green.png")
        green_png(mask, layer_png)
        above = layer_below_text(layers)
        out = os.path.join(out_dir, os.path.basename(psd))
        os.makedirs(out_dir, exist_ok=True)
        run(["node", "--max-old-space-size=6144", os.path.join(HERE, "add_raster_layer.mjs"), psd, layer_png, out,
             "--name", a.name, "--above", above, "--left", str(orig["left"]), "--top", str(orig["top"])])
    share = 100.0 * (mask > 0).mean()
    print(f"{stem}: green {share:.2f}% of the page, text over art: {'yes' if n_text else 'no'}, gutter: {gutter}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("psd", nargs="+")
    ap.add_argument("--output-dir")
    ap.add_argument("--name", default="Green screen")
    ap.add_argument("--grow", type=int, default=3, help="px added around the detected strokes (the lines around the letters)")
    ap.add_argument("--no-gutter", action="store_true", help="skip the middle-of-the-spread detection")
    a = ap.parse_args()
    root = os.environ.get("COMIC_OUTPUT_DIR") or os.path.expanduser("~/Downloads")
    default = os.path.join(root, os.path.basename(os.path.dirname(os.path.abspath(a.psd[0]))) + " green")
    out_dir = a.output_dir or default
    rc = 0
    for psd in a.psd:
        try:
            process(psd, out_dir, a)
        except Exception as e:
            print(f"FAIL {psd}: {e}", file=sys.stderr); rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
