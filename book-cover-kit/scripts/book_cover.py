#!/usr/bin/env python3
"""book-cover-kit CLI: phone photo of a book -> flat cover image, shelf colours,
ISBN from the barcode. Works on one file or every image in a folder.

  flatten <photo|folder>  warp the cover flat, save <stem>.jpg (max --max-side px)
  palette <image|folder>  spine / text / accent colours as JSON
  barcode <photo|folder>  EAN-13 / ISBN found on the image as JSON
  all <photo|folder>      flatten + palette + barcode, one JSON record per image
  shrink <image|folder>   no geometry: just resize to --max-side and re-save as JPEG
                          (catalogue thumbnails of already-flat covers, e.g. Calibre's)

Results go to --output (default ~/Downloads/<source name> covers, or
$COMIC_OUTPUT_DIR instead of ~/Downloads); the input is never modified.

Example:
    venv/bin/python book-cover-kit/scripts/book_cover.py all ~/Pictures/new-books --output /tmp/covers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cover_barcode import read_barcodes  # noqa: E402
from cover_geometry import flatten_cover  # noqa: E402
from cover_palette import cover_palette  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise SystemExit(f"not a file or folder: {source}")
    return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def default_output(source: Path) -> Path:
    root = Path(os.environ.get("COMIC_OUTPUT_DIR") or Path.home() / "Downloads")
    name = source.stem if source.is_file() else source.name
    return root / f"{name} covers"


def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read image {path}: expected jpg/png/webp/tif")
    return img


def shrink(img, max_side: int):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def process(path: Path, command: str, output: Path, max_side: int) -> dict:
    img = load_image(path)
    record: dict = {"source": str(path)}
    if command == "shrink":
        small = shrink(img, max_side)
        target = output / f"{path.stem}.jpg"
        cv2.imwrite(str(target), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        record.update(cover=str(target), size=[small.shape[1], small.shape[0]])
    if command in ("flatten", "all"):
        flat, quad = flatten_cover(img)
        flat = shrink(flat, max_side)
        target = output / f"{path.stem}.jpg"
        cv2.imwrite(str(target), flat, [cv2.IMWRITE_JPEG_QUALITY, 90])
        record.update(cover=str(target), warped=quad is not None, size=[flat.shape[1], flat.shape[0]])
        img = flat
    if command in ("palette", "all"):
        record["palette"] = cover_palette(img)
    if command in ("barcode", "all"):
        record["barcodes"] = read_barcodes(load_image(path))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["flatten", "palette", "barcode", "all", "shrink"])
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, help="folder for covers + results.json")
    parser.add_argument("--max-side", type=int, default=1400, help="longest side of a saved cover (px)")
    args = parser.parse_args(argv)
    source = args.source.expanduser().resolve()
    output = (args.output or default_output(source)).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = [process(p, args.command, output, args.max_side) for p in list_images(source)]
    (output / "results.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(records, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
