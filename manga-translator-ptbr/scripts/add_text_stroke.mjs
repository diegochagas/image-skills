#!/usr/bin/env node
// Gives every white text layer an outside BLACK stroke and every black text
// layer an outside WHITE stroke (Photoshop Layer Style > Stroke, position
// Outside), so the lettering stays readable over art. Small and medium text
// gets 3 px, big text 6 px; "big" = font size >= --big-from (default 60 px).
// Text in any other colour is left alone and listed. Re-running replaces the
// stroke it wrote, so the script is idempotent. Rasters stay byte-exact.
//
// Usage: node add_text_stroke.mjs <psd | folder> [--from STEM] [--to STEM]
//            [--only STEM ...] [--small 3] [--big 6] [--big-from 60]
//            [--output DIR | --in-place] [--dry-run]
//   --from/--to  inclusive range of stems by page number (063 .. 235; a
//                spread such as 072-073 counts by its first number)
//   --output     default ~/Downloads/<folder name> stroke/ (COMIC_OUTPUT_DIR
//                replaces ~/Downloads); --in-place rewrites the inputs (back
//                them up first)
import * as fs from 'fs';
import * as path from 'path';
import { readPsd, writePsdBuffer, initializeCanvas } from 'ag-psd';
import { createCanvas, createImageData } from 'canvas';
import { normalizeTextLayer } from './fit_boxes.mjs';
import { compositeFromTopRaster } from './psd_preview.mjs';

initializeCanvas(createCanvas, createImageData);

const WHITE = { r: 255, g: 255, b: 255 };
const BLACK = { r: 0, g: 0, b: 0 };

/** Stroke colour for a text fill colour: black on white text, white on black text, null for others. */
export function strokeColorFor(fill) {
  if (!fill) return null;
  const { r, g, b } = fill;
  if (r >= 250 && g >= 250 && b >= 250) return BLACK;
  if (r <= 5 && g <= 5 && b <= 5) return WHITE;
  return null;
}

/** Stroke width in px: bigPx for font sizes >= bigFrom, smallPx otherwise. */
export function strokeWidthFor(fontSize, { smallPx, bigPx, bigFrom }) {
  return fontSize >= bigFrom ? bigPx : smallPx;
}

/** Photoshop outside-stroke effect (ag-psd shape) of the given colour and width. */
export function outsideStroke(color, sizePx) {
  return {
    enabled: true, present: true, showInDialog: true, size: { value: sizePx, units: 'Pixels' },
    position: 'outside', fillType: 'color', blendMode: 'normal', opacity: 1, color: { ...color },
  };
}

/** Applies the stroke to one text layer; returns a short description, or null when the colour is skipped. */
export function strokeTextLayer(layer, widths) {
  const fill = layer.text.style && layer.text.style.fillColor;
  const color = strokeColorFor(fill);
  if (!color) return null;
  const size = layer.text.style.fontSize;
  const px = strokeWidthFor(size, widths);
  layer.effects = { ...(layer.effects || {}), stroke: [outsideStroke(color, px)] };
  return `${layer.name}: ${size}px ${color.r ? 'white' : 'black'} text -> ${px}px ${color.r ? 'black' : 'white'} stroke`;
}

function main() {
  const args = process.argv.slice(2);
  const opt = (n, d) => { const i = args.indexOf(n); return i !== -1 ? args[i + 1] : d; };
  const src = args[0];
  if (!src || src.startsWith('--')) { console.error('usage: add_text_stroke.mjs <psd|folder> [--from STEM] [--to STEM] [--only STEM ...] [--small 3] [--big 6] [--big-from 60] [--output DIR | --in-place] [--dry-run]'); process.exit(1); }
  const widths = { smallPx: Number(opt('--small', 3)), bigPx: Number(opt('--big', 6)), bigFrom: Number(opt('--big-from', 60)) };
  const from = opt('--from', null), to = opt('--to', null);
  const onlyI = args.indexOf('--only');
  const only = onlyI === -1 ? null : new Set(args.slice(onlyI + 1).filter((a) => !a.startsWith('--')));
  const inPlace = args.includes('--in-place'), dry = args.includes('--dry-run');
  const stat = fs.statSync(src);
  const files = stat.isDirectory() ? fs.readdirSync(src).filter((n) => /\.psd$/i.test(n)).sort().map((n) => path.join(src, n)) : [src];
  const root = process.env.COMIC_OUTPUT_DIR || path.join(process.env.HOME || '', 'Downloads');
  const outDir = inPlace ? null : (opt('--output', null) || path.join(root, `${path.basename(stat.isDirectory() ? src : path.dirname(src))} stroke`));
  const key = (f) => path.basename(f, path.extname(f));
  const cmp = (a, b) => a.localeCompare(b, undefined, { numeric: true });
  const selected = files.filter((f) => (!from || cmp(key(f), from) >= 0) && (!to || parseInt(key(f), 10) <= parseInt(to, 10)) && (!only || only.has(key(f))));
  console.log(`${selected.length} PSD(s)${dry ? ' (dry run)' : inPlace ? ' (in place)' : ` -> ${outDir}`}`);
  let done = 0, skipped = 0;
  for (const f of selected) {
    const buf = fs.readFileSync(f);
    const psd = readPsd(buf, { useRawData: true, skipCompositeImageData: true, skipThumbnail: true });
    const notes = [];
    for (const l of psd.children || []) {
      if (!l.text) continue;
      const note = strokeTextLayer(l, widths);
      if (note) notes.push(note); else { skipped++; console.log(`${path.basename(f)}: ${l.name} left alone (fill ${JSON.stringify(l.text.style && l.text.style.fillColor)} is neither white nor black)`); }
      if (note) normalizeTextLayer(l);
    }
    console.log(`${path.basename(f)}: ${notes.length} stroke(s)`);
    if (!notes.length || dry) continue;
    const out = inPlace ? f : path.join(outDir, path.basename(f));
    psd.imageData = compositeFromTopRaster(buf, psd.width, psd.height);
    fs.mkdirSync(path.dirname(out), { recursive: true });
    fs.writeFileSync(out, writePsdBuffer(psd, { generateThumbnail: false }));
    done++;
  }
  console.log(`done: ${done} file(s) ${dry ? 'would change' : 'written'}, ${skipped} text layer(s) left alone`);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
