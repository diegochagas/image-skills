#!/usr/bin/env python3
"""Add a "text on solid colours erased, cut with margin" layer to finished PSDs.

    add_solid_layer.py <raw scan> <cleaned scan> <psd> [<psd> ...] [--margin 0.15]
                       [--name NAME] [--output-dir DIR | --in-place] [--above LAYER]

<cleaned scan> is the raw scan after detect_blocks.py + clean_blocks.py (text
erased ONLY over single colours; strokes over art stay). Each PSD's bottom
layer ("Original") is located in the raw scan (template match, exact to the
pixel), and the cleaned scan is cut at the same place PLUS `--margin` (a
fraction of the page width, 0.15 = about 2.5 cm) of the neighbouring page on
the side of the cut, so the cut can be adjusted by hand. The canvas grows by
that strip (every other layer keeps its place; a strip on the left moves
them right). A PSD that already holds the whole scan (a spread) is not cut:
it gets the cleaned scan as is.

Example: add_solid_layer.py scan.jpg scan_cleaned.png 063.psd --in-place
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = "node"
COARSE = 0.2
REFINE_PX = 12


def export_original(psd, png):
    subprocess.run([NODE, "--max-old-space-size=4096", os.path.join(HERE, "export_layer.mjs"), psd, "0", png],
                   check=True, stdout=subprocess.DEVNULL)


def locate(scan_gray, page_gray):
    """(x, y, score): where `page_gray` sits in `scan_gray`, exact to the pixel."""
    sh, sw = scan_gray.shape
    ph, pw = page_gray.shape
    pad = 40
    small = cv2.resize(scan_gray, None, fx=COARSE, fy=COARSE, interpolation=cv2.INTER_AREA)
    small = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    tpl = cv2.resize(page_gray, None, fx=COARSE, fy=COARSE, interpolation=cv2.INTER_AREA)
    r = cv2.matchTemplate(small, tpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(r)
    cx, cy = int(round((loc[0] - pad) / COARSE)), int(round((loc[1] - pad) / COARSE))
    # refine at full resolution on the page's central band, search window +-REFINE_PX
    band = page_gray[ph // 4: 3 * ph // 4, pw // 8: 7 * pw // 8]
    bx, by = pw // 8, ph // 4
    best = (-1.0, cx, cy)
    x0, y0 = max(0, cx + bx - REFINE_PX), max(0, cy + by - REFINE_PX)
    x1, y1 = min(sw, cx + bx + band.shape[1] + REFINE_PX), min(sh, cy + by + band.shape[0] + REFINE_PX)
    win = scan_gray[y0:y1, x0:x1]
    if win.shape[0] >= band.shape[0] and win.shape[1] >= band.shape[1]:
        rr = cv2.matchTemplate(win, band, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(rr)
        best = (float(mx), x0 + ml[0] - bx, y0 + ml[1] - by)
    return best[1], best[2], best[0]


def cut_plan(x, y, pw, ph, sw, sh, margin):
    """Crop box (x0, y0, x1, y1) in the scan, columns added left / right.
    A margin is added only on a side that is a cut (not the scan's own edge)."""
    m = int(round(margin * pw))
    tol = 0.05 * pw
    left_cut = x > tol
    right_cut = (sw - (x + pw)) > tol
    # a page touching the scan's edge on one side is the half cut on the other side
    ml = m if left_cut else 0
    mr = m if right_cut else 0
    ml, mr = min(ml, x), min(mr, sw - (x + pw))
    return (x - ml, y, x + pw + mr, y + ph), ml, mr


def process(scan, cleaned, psd, a, work):
    orig_png = os.path.join(work, "orig.png")
    export_original(psd, orig_png)
    page = cv2.imread(orig_png, cv2.IMREAD_GRAYSCALE)
    x, y, score = locate(cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY), page)
    ph, pw = page.shape
    sh, sw = scan.shape[:2]
    if score < 0.9 or y < 0 or y + ph > sh:
        raise RuntimeError(f"{psd}: Original not found in the scan (score {score:.3f}, at {x},{y}, page {pw}x{ph}, scan {sw}x{sh})")
    (cx0, cy0, cx1, cy1), ml, mr = cut_plan(x, y, pw, ph, sw, sh, a.margin)
    layer_png = os.path.join(work, "layer.png")
    cv2.imwrite(layer_png, cleaned[cy0:cy1, cx0:cx1])
    if a.in_place:
        out = psd
    else:
        out = os.path.join(a.output_dir, os.path.basename(psd))
    tmp = os.path.join(work, "out.psd")
    existing = subprocess.run([NODE, os.path.join(HERE, "list_layers.mjs"), psd], capture_output=True, text=True, check=True).stdout
    above = a.above or ("Qwen clean" if '"Qwen clean"' in existing else "Copy")
    res = subprocess.run([NODE, "--max-old-space-size=6144", os.path.join(HERE, "add_cut_layer.mjs"), psd, layer_png, tmp,
                          "--name", a.name, "--above", above, "--width", str(pw + ml + mr), "--shift-x", str(ml)],
                         capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f"{psd}: {res.stderr.strip()}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.move(tmp, out)
    print(f"{os.path.basename(psd)}: page at {x},{y} (match {score:.4f}), margin left {ml} right {mr} -> {pw + ml + mr}x{ph}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan")
    ap.add_argument("cleaned")
    ap.add_argument("psd", nargs="+")
    ap.add_argument("--margin", type=float, default=0.15, help="fraction of the page width kept past the cut")
    ap.add_argument("--name", default="Solid text erased + cut margin")
    ap.add_argument("--above", help="layer the new one goes above (default: Qwen clean if present, else Copy)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--output-dir")
    g.add_argument("--in-place", action="store_true")
    a = ap.parse_args()
    scan, cleaned = cv2.imread(a.scan), cv2.imread(a.cleaned)
    if scan is None or cleaned is None or scan.shape != cleaned.shape:
        print("cannot read the scan / cleaned image, or their sizes differ", file=sys.stderr)
        return 1
    rc = 0
    for psd in a.psd:
        with tempfile.TemporaryDirectory(prefix="solidlayer_") as work:
            try:
                process(scan, cleaned, psd, a, work)
            except Exception as e:
                print(f"FAIL {e}", file=sys.stderr); rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
