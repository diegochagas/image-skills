#!/usr/bin/env python3
"""Repair the middle of a double-page scan with Nano Banana Pro, changing pixels ONLY inside the gutter mask.

As a module: `repair_gutter(page_bgr, mask, seam, work_dir, cost=False) -> (image, generations)`.
As a CLI (test one page image): gutter_ai.py <spread.png> --out DIR [--cost]

How (worked out on a Saint Seiya spread, 2026-11-20):
  1. The page is cut into overlapping 9:16 crops centred on the seam line
     (gutter_mask.find_seam), wide enough that the sharp art on both sides is
     in the crop; each crop is upscaled to the model's 1152x2048 and sent to
     Higgsfield `nano_banana_pro` (2 credits) with GUTTER_PROMPT.
  2. The answer is registered back onto its crop (affine ECC on the area
     outside the mask: the model rescales by ~0.8 % and shifts ~1 px), its
     tones are matched to the scan on a ring just outside the mask, and the
     crops are blended where they overlap.
  3. Only inside the gutter mask (feathered) the model's pixels replace the
     scan's; everything else stays the scan.
  4. The model sometimes keeps a 1 px trace of the seam line: in the seam's
     own columns a pixel clearly darker than both neighbours 2 px away is
     replaced by their blend (TRACE_DROP).
Each crop's raw answer is kept in work_dir (crop_<i>.png / gen_<i>.png) for QC.

Example: gutter_ai.py ~/Downloads/x/072-073.png --out /tmp/gutter_ai
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gutter_mask  # noqa: E402

MODEL = "nano_banana_pro"
UPLOAD = (1152, 2048)        # 9:16 at the model's 2k
CROP_W_FRAC = 0.14           # crop width as a fraction of the spread width (sharp art on both sides)
OVERLAP = 0.15               # crops overlap by this share of their height
FEATHER = 6                  # px of soft edge on the mask
RING = (8, 20)               # tone-match ring: this many px outside the mask
TRACE_DROP = 20              # a seam pixel this much darker than both neighbours is a trace
GUTTER_PROMPT = (
    "This is a tall strip cut from a scanned black-and-white manga spread. Down its centre runs the gutter "
    "between the two pages: a thin dark vertical line and, around it, a narrow band where the drawing is out "
    "of focus or washed out. Repair ONLY that central band: remove the dark vertical line and redraw the "
    "blurred lines, starburst spikes, stars, outlines and stripes there sharply, so every drawn element "
    "continues seamlessly from the left half into the right half, in the same ink style, line widths and grey "
    "tones as the sharp parts of the drawing. This is a repair, not a new picture: keep every line exactly "
    "where it is, keep the composition, the paper grain and the framing; do not add, move or restyle "
    "anything. Black and white only. No text, no borders.")

Seam = Tuple[float, float, float]


def seam_x(seam: Seam, y: np.ndarray, h: int) -> np.ndarray:
    """x of the seam line at rows y (gutter_mask's convention: x at mid height + tilt over the height)."""
    return seam[0] + seam[1] * (y / max(1, h - 1) - 0.5)


def plan_crops(h: int, w: int, seam: Seam) -> List[Dict[str, int]]:
    """Overlapping 9:16 crops down the seam. Example: 3599x2548 spread -> 4 crops of 504x896."""
    cw = int(round(w * CROP_W_FRAC / 2)) * 2
    ch = min(h, int(round(cw * 16 / 9)))
    n = max(1, int(np.ceil((h - ch * OVERLAP) / (ch * (1 - OVERLAP)))))
    tops = np.linspace(0, h - ch, n).round().astype(int) if n > 1 else [0]
    crops = []
    for i, y0 in enumerate(tops):
        xc = int(round(float(seam_x(seam, np.array([y0 + ch / 2]), h)[0])))
        crops.append(dict(i=i, x0=int(np.clip(xc - cw // 2, 0, w - cw)), y0=int(y0), w=cw, h=ch))
    return crops


def run_cli(cmd: List[str]) -> str:
    """Higgsfield CLI stdout; stops on auth errors, retries transient ones."""
    err = ""
    for attempt in range(3):
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        err = (res.stderr or "") + (res.stdout or "")
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout
        if re.search(r"unauthori[sz]ed|not logged in|auth login|\b401\b", err, re.I):
            raise SystemExit(f"higgsfield: not logged in - run `higgsfield auth login` ({err[:200]})")
    raise RuntimeError(f"higgsfield failed: {err[-400:]}")


def result_url(answer: str) -> str:
    """Full-size result URL in the CLI's JSON (not the *_min.webp preview, not the uploaded input)."""
    urls = re.findall(r'https://[^"\s]+/hf_[^"\s]+?\.(?:png|jpg|jpeg|webp)', answer)
    full = [u for u in urls if "_min." not in u]
    if not full:
        raise RuntimeError(f"no image in the answer: {answer[:300]}")
    return full[0]


def generate(crop: np.ndarray, path_in: str, cost: bool, prompt: str = GUTTER_PROMPT,
             aspect: str = "9:16", size: Tuple[int, int] = UPLOAD) -> Optional[np.ndarray]:
    """One Nano Banana Pro edit of a crop (None with cost=True: price only). The crop is uploaded at `size`."""
    cv2.imwrite(path_in, cv2.resize(crop, size, interpolation=cv2.INTER_LANCZOS4))
    hf = shutil.which("higgsfield") or "higgsfield"
    cmd = [hf, "generate", "cost" if cost else "create", MODEL, "--prompt", prompt,
           "--aspect_ratio", aspect, "--resolution", "2k", "--image-references", path_in]
    if cost:
        print(f"  crop price: {run_cli(cmd).strip()}")
        return None
    answer = run_cli(cmd + ["--json", "--wait", "--wait-timeout", "12m"])
    raw = path_in.replace("crop_", "gen_")
    urllib.request.urlretrieve(result_url(answer), raw)
    gen = cv2.imread(raw)
    if gen is None:
        raise RuntimeError(f"unreadable model answer {raw}")
    return gen


def register(gen: np.ndarray, crop: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """The model's answer warped onto the crop (affine ECC on `keep` = area outside the mask), crop size."""
    h, w = crop.shape[:2]
    s = 4
    g = cv2.resize(gen, (w * s, h * s), interpolation=cv2.INTER_AREA).astype(np.float32)
    t = cv2.resize(crop, (w * s, h * s), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    k = (cv2.resize(keep.astype(np.uint8), (w * s, h * s), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        _, warp = cv2.findTransformECC(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY), cv2.cvtColor(g, cv2.COLOR_BGR2GRAY),
                                       warp, cv2.MOTION_AFFINE, crit, k, 5)
    except cv2.error:
        pass                                    # no convergence: plain resize (scale error ~1 %)
    reg = cv2.warpAffine(g, warp, (w * s, h * s), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REFLECT)
    return cv2.resize(reg, (w, h), interpolation=cv2.INTER_AREA)


def match_tones(reg: np.ndarray, crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel linear fit of the answer onto the scan, on a ring just outside the mask."""
    m = (mask > 0).astype(np.uint8)
    ring = (cv2.dilate(m, np.ones((2 * RING[1] + 1,) * 2, np.uint8)) > 0) & \
           ~(cv2.dilate(m, np.ones((2 * RING[0] + 1,) * 2, np.uint8)) > 0)
    if ring.sum() < 50:
        return reg
    out = reg.copy()
    for c in range(3):
        a, b = np.polyfit(reg[..., c][ring], crop[..., c][ring].astype(np.float32), 1)
        out[..., c] = reg[..., c] * a + b
    return out


def remove_trace(img: np.ndarray, seam: Seam, mask: np.ndarray) -> int:
    """Blend away a 1 px seam trace the model kept (in place). Returns the pixels changed."""
    h = img.shape[0]
    g = img.mean(axis=2)
    n = 0
    for y in np.nonzero(mask.any(axis=1))[0]:
        xc = float(seam_x(seam, np.array([y]), h)[0])
        for x in range(int(np.floor(xc - 1)), int(np.ceil(xc + 2)) + 1):
            if 2 <= x < img.shape[1] - 2 and g[y, x] < min(g[y, x - 2], g[y, x + 2]) - TRACE_DROP:
                img[y, x] = (img[y, x - 2] + img[y, x + 2]) / 2
                n += 1
    return n


def repair_gutter(page: np.ndarray, mask: np.ndarray, seam: Seam, work: str,
                  cost: bool = False) -> Tuple[np.ndarray, int]:
    """(page with the gutter repaired inside `mask`, number of generations)."""
    os.makedirs(work, exist_ok=True)
    h, w = page.shape[:2]
    crops = plan_crops(h, w, seam)
    if cost:
        generate(page[:crops[0]["h"], :crops[0]["w"]], os.path.join(work, "crop_0.png"), True)
        return page, len(crops)
    acc = np.zeros(page.shape, np.float32)
    wsum = np.zeros((h, w), np.float32)
    for c in crops:
        ys, xs = slice(c["y0"], c["y0"] + c["h"]), slice(c["x0"], c["x0"] + c["w"])
        crop, cm = page[ys, xs], mask[ys, xs]
        gen = generate(crop, os.path.join(work, f"crop_{c['i']}.png"), False)
        reg = match_tones(register(gen, crop, cm == 0), crop, cm)
        fall = np.clip(np.minimum(np.arange(c["h"]), c["h"] - 1 - np.arange(c["h"])) / (c["h"] * OVERLAP / 2), 0.02, 1)
        acc[ys, xs] += reg * fall[:, None, None]
        wsum[ys, xs] += fall[:, None]
    soft = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), FEATHER / 2) * (wsum > 0)
    model = acc / np.maximum(wsum, 1e-6)[..., None]
    out = page.astype(np.float32) * (1 - soft[..., None]) + model * soft[..., None]
    n = remove_trace(out, seam, mask)
    json.dump({"crops": crops, "trace_px": n}, open(os.path.join(work, "gutter_ai.json"), "w"))
    return np.clip(out, 0, 255).astype(np.uint8), len(crops)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cost", action="store_true", help="price only, nothing generated")
    a = ap.parse_args()
    page = cv2.imread(a.image)
    if page is None:
        sys.exit(f"cannot read {a.image}")
    mask, info = gutter_mask.gutter_mask(page)
    if not info["found"]:
        sys.exit(f"no seam found in {a.image}")
    seam = (float(info["x"]), float(info["tilt"]), float(info["score"]))
    out, n = repair_gutter(page, mask, seam, a.out, a.cost)
    if not a.cost:
        cv2.imwrite(os.path.join(a.out, "repaired.png"), out)
    print(f"{os.path.basename(a.image)}: seam {info}, {n} generation(s) x 2 credits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
