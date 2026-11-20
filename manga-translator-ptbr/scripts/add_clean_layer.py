#!/usr/bin/env python3
"""Add an "AI clean" layer to finished PSDs: the page with ALL its text erased, on double pages the middle
repaired, straightened, with Photoshop guides at the page edges - built from the "Solid text erased + cut
margin" layer (--source; the bottom layer "Original" when a PSD has no such layer), placed above every
image layer, under the text boxes.

    add_clean_layer.py <psd> [<psd> ...] [--output-dir DIR | --in-place] [--cost] [--name "AI clean"]
                       [--no-gutter] [--no-text] [--no-align] [--grow 5] [--source LAYER]

Per PSD (on the source layer's pixels):
  1. text on one plain colour: clean_blocks' exact fill (no AI, pixel-perfect);
  2. text over art (a character, screentone, speed lines): text_ai.py - Nano Banana Pro
     on a crop around each block the detector is confident about (2 credits per block;
     blocks only in the cut margin are skipped), used ONLY on the letters clean_blocks
     left untouched, grown by --grow px, registered and tone-matched to the scan;
  3. double page (NNN-MMM or landscape): gutter_ai.py repairs the seam with Nano
     Banana Pro, ONLY inside gutter_mask's band (grown by 1 % of the width, so the
     out-of-focus art next to the seam is redrawn too).
  4. the layer is straightened (page_guides.deskew: the scan's rotation from its long
     straight lines, corners left transparent) and Photoshop guides are added at the
     page edges page_guides.trim_box finds; a side it is not sure about gets no guide
     (the run prints which), so every guide can be trusted and the rest is placed by hand.
Every other pixel is the source layer's. Pages with nothing to do still get the layer.
The two halves of a spread are not shifted against each other: inside the gutter band the
model redraws the lines so they meet.

Output: <DIR>/<psd name> (default ~/Downloads/<source folder> clean, or $COMIC_OUTPUT_DIR/...).
The Higgsfield balance is checked before every page: under --min-credits (40) the run stops
and lists nothing else, so the same command resumes after a top-up (pages that already have
the layer fail with "already has a layer" and are skipped).
--in-place: the PSD itself is replaced (the old file goes to the trash with `gio trash`,
so it can be restored). QC files: <work>/<stem>/ (overlay.jpg: red = AI text erase,
blue = AI gutter repair; the model's raw answers). --cost prints the generations and
their price and generates nothing.

Example: add_clean_layer.py ~/Downloads/Portuguese/080.psd --in-place
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import add_green_layer as green  # noqa: E402
import gutter_ai  # noqa: E402
import gutter_mask  # noqa: E402
import page_guides  # noqa: E402
import text_ai  # noqa: E402

GUTTER_GROW_FRAC = 0.01     # the seam band is widened by this share of the width (blurred art next to it)


def credits_left() -> float:
    """Higgsfield balance from `higgsfield account status` ("... starter plan, 179.5 credits"); inf if unreadable."""
    res = subprocess.run(["higgsfield", "account", "status"], capture_output=True, text=True, timeout=60)
    m = re.search(r"([\d.]+)\s+credits", res.stdout)
    return float(m.group(1)) if m else float("inf")


def out_root() -> str:
    return os.environ.get("COMIC_OUTPUT_DIR") or os.path.expanduser("~/Downloads")


def text_blocks(tmp: str, stem: str, layers: list, src: dict) -> list:
    """Confident text blocks of the source layer, minus those only in the cut margin (outside the Original)."""
    base0 = layers[0]
    keep_x = (base0["left"] - src["left"], base0["left"] - src["left"] + base0["width"])
    det = os.path.join(tmp, "detect", f"{stem}_detect.json")
    return text_ai.confident_blocks(det, det.replace("_detect.json", "_raw.json"), keep_x)


def gutter_band(page: np.ndarray) -> tuple:
    """(band mask, seam) of a double page, or (None, None) when no seam shows."""
    mask, info = gutter_mask.gutter_mask(page)
    if not info["found"]:
        return None, None
    g = max(3, int(page.shape[1] * GUTTER_GROW_FRAC))
    mask = cv2.dilate(mask, np.ones((1, 2 * g + 1), np.uint8))
    return mask, (float(info["x"]), float(info["tilt"]), float(info["score"]))


def overlay(page: np.ndarray, art: np.ndarray, band) -> np.ndarray:
    out = page.copy()
    out[art > 0] = (0.5 * out[art > 0] + 0.5 * np.array((0, 0, 255))).astype(np.uint8)
    if band is not None:
        out[band > 0] = (0.6 * out[band > 0] + 0.4 * np.array((255, 0, 0))).astype(np.uint8)
    return out


def source_layer(layers: list, name: str) -> dict:
    """The layer the cleaning starts from: `name` if the PSD has it, else the Original (bottom layer)."""
    return next((l for l in layers if l["name"] == name), layers[0])


def guides_of(box: dict, src: dict) -> str:
    """add_raster_layer's --guides value in document px. Example: {"left": 460, "top": None, ...} -> "v:460,..."."""
    parts = [f"v:{src['left'] + box[k]}" for k in ("left", "right") if box[k] is not None]
    parts += [f"h:{src['top'] + box[k]}" for k in ("top", "bottom") if box[k] is not None]
    return ",".join(parts)


def build_layer(psd: str, tmp: str, work: str, a: argparse.Namespace) -> tuple:
    """(path of the layer PNG, report dict) for one PSD."""
    stem = Path(psd).stem
    layers = green.layers_of(psd)
    orig = source_layer(layers, a.source)
    page_png = os.path.join(tmp, f"{stem}.png")
    green.run(["node", "--max-old-space-size=4096", os.path.join(HERE, "export_layer.mjs"), psd,
               str(orig["index"]), page_png])
    art = green.art_text_mask(page_png, tmp, stem) if not a.no_text else np.zeros((orig["height"], orig["width"]), np.uint8)
    base = cv2.imread(os.path.join(tmp, "detect", f"{stem}_cleaned.png")) if not a.no_text else None
    page = cv2.imread(page_png)
    base = page if base is None else base
    art = cv2.dilate(art, np.ones((2 * a.grow + 1,) * 2, np.uint8)) if art.any() else art
    gens = {"text": 0, "gutter": 0}
    blocks = text_blocks(tmp, stem, layers, orig) if art.any() else []
    used = np.zeros_like(art)
    for x, y, w, h in blocks:
        used[y:y + h, x:x + w] = art[y:y + h, x:x + w]
    art = used
    if blocks:
        base, gens["text"] = text_ai.erase_art_text(base, art, blocks, os.path.join(work, "text"), a.cost)
    band = None
    if not a.no_gutter and green.is_double(stem, orig["width"], orig["height"]):
        band, seam = gutter_band(page)
        if band is not None:
            base, gens["gutter"] = gutter_ai.repair_gutter(base, band, seam, os.path.join(work, "gutter"), a.cost)
    cv2.imwrite(os.path.join(work, "overlay.jpg"), overlay(page, art, band), [cv2.IMWRITE_JPEG_QUALITY, 85])
    # tilt and page edges are measured on the source as it was (erasing text changes its long lines)
    angle = 0.0 if a.no_align else page_guides.deskew(page)[1]
    straight = page_guides.rotate(base, angle)
    box = page_guides.trim_box(page_guides.rotate(page, angle, transparent=False))
    layer_png = os.path.join(tmp, "clean.png")
    cv2.imwrite(layer_png, straight)
    return layer_png, dict(stem=stem, layers=layers, orig=orig, gens=gens, angle=angle, box=box,
                           guides=guides_of(box, orig), art=float((art > 0).mean()), gutter=band is not None)


def write_psd(psd: str, layer_png: str, rep: dict, dest: str, name: str) -> None:
    above = green.layer_below_text(rep["layers"])
    green.run(["node", "--max-old-space-size=6144", os.path.join(HERE, "add_raster_layer.mjs"), psd, layer_png, dest,
               "--name", name, "--above", above, "--left", str(rep["orig"]["left"]), "--top", str(rep["orig"]["top"])]
              + (["--guides", rep["guides"]] if rep["guides"] else []))


def place(psd: str, tmp_out: str, a: argparse.Namespace, out_dir: str) -> str:
    """Move the written PSD to its destination; --in-place trashes the old file first."""
    if not a.in_place:
        dest = os.path.join(out_dir, os.path.basename(psd))
        shutil.move(tmp_out, dest)
        return dest
    subprocess.run(["gio", "trash", psd], check=True)
    shutil.move(tmp_out, psd)
    return psd


def process(psd: str, out_dir: str, work_root: str, a: argparse.Namespace) -> dict:
    stem = Path(psd).stem
    work = os.path.join(work_root, stem)
    os.makedirs(work, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clean_layer_") as tmp:
        layer_png, rep = build_layer(psd, tmp, work, a)
        if a.cost:
            return rep
        if any(l["name"] == a.name for l in rep["layers"]):
            raise RuntimeError(f'{stem} already has a layer named "{a.name}"')
        tmp_out = os.path.join(os.path.dirname(os.path.abspath(psd)) if a.in_place else out_dir, f".{stem}.clean.psd")
        write_psd(psd, layer_png, rep, tmp_out, a.name)
        rep["dest"] = place(psd, tmp_out, a, out_dir)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("psd", nargs="+")
    ap.add_argument("--output-dir")
    ap.add_argument("--in-place", action="store_true", help="replace the PSDs (old files go to the trash)")
    ap.add_argument("--name", default="AI clean")
    ap.add_argument("--grow", type=int, default=5, help="px around the text-over-art strokes taken from the AI erase")
    ap.add_argument("--no-gutter", action="store_true")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--no-align", action="store_true", help="do not straighten the layer")
    ap.add_argument("--source", default="Solid text erased + cut margin",
                    help="layer the cleaning starts from (falls back to the bottom layer)")
    ap.add_argument("--cost", action="store_true", help="count generations and print prices, generate nothing")
    ap.add_argument("--min-credits", type=float, default=40, help="stop before a page when the balance is below this")
    a = ap.parse_args()
    folder = os.path.basename(os.path.dirname(os.path.abspath(a.psd[0])))
    out_dir = a.output_dir or os.path.join(out_root(), f"{folder} clean")
    work_root = os.path.join(out_root(), f"{folder} clean-layer-work")
    if not a.in_place:
        os.makedirs(out_dir, exist_ok=True)
    rc, total = 0, 0
    for psd in a.psd:
        left = float("inf") if a.cost else credits_left()
        if left < a.min_credits:
            print(f"STOP before {Path(psd).stem}: {left:g} credits left (< {a.min_credits:g}); "
                  "the rest of the list was not touched", flush=True)
            return 3
        try:
            rep = process(psd, out_dir, work_root, a)
            n = rep["gens"]["text"] + rep["gens"]["gutter"]
            total += n
            print(f"{rep['stem']}: text over art {rep['art']:.2%} of the page, gutter {'yes' if rep['gutter'] else 'no'}, "
                  f"generations text {rep['gens']['text']} + gutter {rep['gens']['gutter']}, "
                  f"skew {rep['angle']:+.2f} deg, guides {rep['guides'] or 'none'}"
                  + (f" (no guide: {', '.join(k for k, v in rep['box'].items() if v is None)})" if None in rep['box'].values() else "")
                  + ("" if a.cost else f" -> {rep['dest']}"), flush=True)
        except Exception as e:  # one bad page must not stop the folder
            print(f"FAIL {psd}: {e}", file=sys.stderr, flush=True)
            rc = 2
    print(f"total generations: {total} x 2 credits (Nano Banana Pro)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
