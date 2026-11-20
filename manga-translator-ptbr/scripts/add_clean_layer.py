#!/usr/bin/env python3
"""Add a "Clean" layer to finished PSDs: on double pages the middle repaired (gutter_local.py: halves
aligned, spine veil/shadow and the dividing line removed - free, no AI), the page straightened, with
Photoshop guides at the page edges. Built from the "Solid text erased + cut margin" layer (--source; the
bottom layer "Original" when a PSD has none), placed above every image layer, under the text boxes.

    add_clean_layer.py <psd> [<psd> ...] [--output-dir DIR | --in-place] [--name "Clean"]
                       [--no-gutter] [--no-align] [--source LAYER]

Per PSD (on the source layer's pixels):
  1. double page (NNN-MMM or landscape) with a seam gutter_mask finds: gutter_local.repair_gutter;
  2. the layer is straightened (page_guides.deskew: the scan's rotation from its long straight
     lines, measured on the source, corners left transparent) and Photoshop guides are added at
     the page edges page_guides.trim_box finds; a side it is not sure of gets no guide (the run
     prints which), so every guide can be trusted and the rest is placed by hand.
Text over art is NOT erased (no free method was good enough; the Green screen layer marks it).
Every other pixel is the source layer's. Pages with nothing to repair still get the layer.

Output: <DIR>/<psd name> (default ~/Downloads/<source folder> clean, or $COMIC_OUTPUT_DIR/...).
--in-place: the PSD itself is replaced (the old file goes to the trash with `gio trash`). A PSD
that already has a layer named --name is refused, so a rerun of the same list resumes;
--replace rebuilds that layer instead.
QC: <work>/<stem>_gutter.jpg (before | after strip of each repaired seam).

Example: add_clean_layer.py ~/Downloads/Portuguese/*.psd --in-place
"""
import argparse
import os
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
import gutter_local  # noqa: E402
import gutter_mask  # noqa: E402
import page_guides  # noqa: E402

CENTRE_FRAC = 0.12          # the spine is looked for within +-12 % of the width from the centre


def out_root() -> str:
    return os.environ.get("COMIC_OUTPUT_DIR") or os.path.expanduser("~/Downloads")


def source_layer(layers: list, name: str) -> dict:
    """The layer the cleaning starts from: `name` if the PSD has it, else the Original (bottom layer)."""
    return next((l for l in layers if l["name"] == name), layers[0])


def guides_of(box: dict, src: dict) -> str:
    """add_raster_layer's --guides value in document px. Example: {"left": 460, "top": None, ...} -> "v:460,..."."""
    parts = [f"v:{src['left'] + box[k]}" for k in ("left", "right") if box[k] is not None]
    parts += [f"h:{src['top'] + box[k]}" for k in ("top", "bottom") if box[k] is not None]
    return ",".join(parts)


def centre_seam(page: np.ndarray) -> tuple:
    """gutter_mask's seam, searched only in the middle CENTRE_FRAC of the width: a spread's spine is
    near its centre, and a wider search picks panel borders (164-165, 206-207, 232-233). None if absent."""
    w = page.shape[1]
    x0 = int(w * (0.5 - CENTRE_FRAC))
    gray = cv2.cvtColor(page[:, x0:w - x0], cv2.COLOR_BGR2GRAY)
    seam = gutter_mask.find_seam(gray)
    return None if seam is None else (seam[0] + x0, seam[1], seam[2])


def repair(page: np.ndarray, stem: str, src: dict, work: str, a: argparse.Namespace) -> tuple:
    """(page with the seam repaired, info or None) - only for a double page whose seam shows."""
    if a.no_gutter or not green.is_double(stem, src["width"], src["height"]):
        return page, None
    seam = centre_seam(page)
    if seam is None:
        return page, {"seam": "not found"}
    out, info = gutter_local.repair_gutter(page, seam)
    x, r = int(info["x_mid"]), int(60 * page.shape[1] / gutter_local.REF_W)
    strip = np.hstack([page[:, x - r:x + r], np.full((page.shape[0], 6, 3), 255, np.uint8), out[:, x - r:x + r]])
    cv2.imwrite(os.path.join(work, f"{stem}_gutter.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out, info


def build_layer(psd: str, tmp: str, work: str, a: argparse.Namespace) -> tuple:
    """(path of the layer PNG, report dict) for one PSD."""
    stem = Path(psd).stem
    layers = green.layers_of(psd)
    src = source_layer(layers, a.source)
    page_png = os.path.join(tmp, f"{stem}.png")
    green.run(["node", "--max-old-space-size=4096", os.path.join(HERE, "export_layer.mjs"), psd,
               str(src["index"]), page_png])
    page = cv2.imread(page_png)
    base, gutter = repair(page, stem, src, work, a)
    angle = 0.0 if a.no_align else page_guides.deskew(page)[1]
    box = page_guides.trim_box(page_guides.rotate(page, angle, transparent=False))
    layer_png = os.path.join(tmp, "clean.png")
    cv2.imwrite(layer_png, page_guides.rotate(base, angle))
    return layer_png, dict(stem=stem, layers=layers, src=src, angle=angle, box=box,
                           guides=guides_of(box, src), gutter=gutter)


def write_psd(psd: str, layer_png: str, rep: dict, dest: str, name: str, replace: bool) -> None:
    layers = [l for l in rep["layers"] if not (replace and l["name"] == name)]
    above = green.layer_below_text(layers)
    green.run(["node", "--max-old-space-size=6144", os.path.join(HERE, "add_raster_layer.mjs"), psd, layer_png, dest,
               "--name", name, "--above", above, "--left", str(rep["src"]["left"]), "--top", str(rep["src"]["top"])]
              + (["--guides", rep["guides"]] if rep["guides"] else []) + (["--replace", "1"] if replace else []))


def place(psd: str, tmp_out: str, a: argparse.Namespace, out_dir: str) -> str:
    """Move the written PSD to its destination; --in-place trashes the old file first."""
    if not a.in_place:
        dest = os.path.join(out_dir, os.path.basename(psd))
        shutil.move(tmp_out, dest)
        return dest
    subprocess.run(["gio", "trash", psd], check=True)
    shutil.move(tmp_out, psd)
    return psd


def process(psd: str, out_dir: str, work: str, a: argparse.Namespace) -> dict:
    stem = Path(psd).stem
    if not a.replace and any(l["name"] == a.name for l in green.layers_of(psd)):
        raise RuntimeError(f'{stem} already has a layer named "{a.name}" (--replace rebuilds it)')
    with tempfile.TemporaryDirectory(prefix="clean_layer_") as tmp:
        layer_png, rep = build_layer(psd, tmp, work, a)
        tmp_out = os.path.join(os.path.dirname(os.path.abspath(psd)) if a.in_place else out_dir, f".{stem}.clean.psd")
        write_psd(psd, layer_png, rep, tmp_out, a.name, a.replace)
        rep["dest"] = place(psd, tmp_out, a, out_dir)
    return rep


def report(rep: dict) -> str:
    missing = [k for k, v in rep["box"].items() if v is None]
    return (f"{rep['stem']}: gutter {rep['gutter'] or 'no'}, skew {rep['angle']:+.2f} deg, "
            f"guides {rep['guides'] or 'none'}" + (f" (no guide: {', '.join(missing)})" if missing else "")
            + f" -> {rep['dest']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("psd", nargs="+")
    ap.add_argument("--output-dir")
    ap.add_argument("--in-place", action="store_true", help="replace the PSDs (old files go to the trash)")
    ap.add_argument("--name", default="Clean")
    ap.add_argument("--no-gutter", action="store_true")
    ap.add_argument("--replace", action="store_true", help="rebuild a layer named --name that the PSD already has")
    ap.add_argument("--no-align", action="store_true", help="do not straighten the layer")
    ap.add_argument("--source", default="Solid text erased + cut margin",
                    help="layer the cleaning starts from (falls back to the bottom layer)")
    a = ap.parse_args()
    folder = os.path.basename(os.path.dirname(os.path.abspath(a.psd[0])))
    out_dir = a.output_dir or os.path.join(out_root(), f"{folder} clean")
    work = os.path.join(out_root(), f"{folder} clean-layer-work")
    os.makedirs(work, exist_ok=True)
    if not a.in_place:
        os.makedirs(out_dir, exist_ok=True)
    rc = 0
    for psd in a.psd:
        try:
            print(report(process(psd, out_dir, work, a)), flush=True)
        except Exception as e:  # one bad page must not stop the folder
            print(f"FAIL {psd}: {e}", file=sys.stderr, flush=True)
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
