---
name: book-cover-kit
description: Turn a phone photo of a book, comic, manga or magazine into what a catalogue needs - the cover warped flat and cropped (perspective from the cover's four edges, table/background removed), the shelf colours of its spine (dominant cover colour, a readable text colour on it, an accent) and the ISBN read from the EAN-13 barcode of a back-cover photo. One image or a whole folder, JSON out. Use when Diego gives cover photos to catalogue (book-menu's add-book-from-cover skill runs it first), asks to "straighten / crop this cover photo", wants a book's ISBN from a photo of its barcode, or needs spine colours for a 3D shelf.
---

# book-cover-kit - cover photo → flat cover, spine colours, ISBN

```bash
<repo>/venv/bin/python book-cover-kit/scripts/book_cover.py <command> <photo-or-folder> [--output DIR] [--max-side 1400]
```

| Command | Output |
| --- | --- |
| `flatten` | `<stem>.jpg`: the cover warped to an upright rectangle (aspect from its own edge lengths), longest side ≤ `--max-side` |
| `palette` | `{"spine": "#rrggbb", "text": "#111111"\|"#f7f3ea", "accent": "#rrggbb", "shares": [...]}` |
| `barcode` | `[{"type": "EAN_13", "value": "978...", "isbn": "978..." or null}]` - `isbn` only for a 978/979 code with a valid check digit |
| `all` | the three above in one record per image (palette computed on the flattened cover) |
| `shrink` | `<stem>.jpg` resized to `--max-side`, nothing else (for covers that are already flat: Calibre, publisher images) |

Every run also writes `results.json` in the output folder. Default output:
`~/Downloads/<source name> covers/` (`$COMIC_OUTPUT_DIR` replaces
`~/Downloads`); the input photos are never modified.

## How it works

- `cover_geometry.py`: the cover is the largest convex four-corner contour
  among three masks (dilated Canny edges, Otsu bright, Otsu dark) that covers
  12-97 % of the photo; it is warped with a homography. No quad (an already
  flat scan, or a cover filling the frame) → only a uniform border is trimmed,
  `"warped": false`.
- `cover_palette.py`: k-means (k=5, fixed seed) on a 96 px copy; spine =
  largest cluster, text = black or off-white by WCAG contrast, accent = most
  saturated cluster covering ≥ 8 %.
- `cover_barcode.py`: OpenCV `BarcodeDetector` on the photo, a sharpened copy
  and three rotations.

## Agent checklist

1. Run `all` on the photo(s). Look at every flattened cover: a wrong crop
   (background left in, a corner cut) means the quad snapped to something
   else (a table edge, the book's spine). Retry on a photo cropped closer, or
   use the photo as is.
2. A back-cover photo is for `barcode` only; never use its flattened image as
   the cover.
3. No barcode found → read the ISBN printed above the bars from the image
   yourself, and say so.

Pure OpenCV/numpy from the shared venv; no model, no network.

Checks: `venv/bin/python -m unittest discover -s book-cover-kit/tests` (synthetic images: angled photo is unwarped to the right aspect, a flat bordered scan is only trimmed, palette, ISBN check digit).
