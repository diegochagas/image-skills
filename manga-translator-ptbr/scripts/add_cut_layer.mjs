#!/usr/bin/env node
// Adds a PNG as a new raster layer of a PSD AND grows the canvas for it: the
// layer is wider than the page (the strip of the neighbouring page kept past
// the cut, so the cut can be adjusted by hand). Every other layer keeps its
// pixels; when the strip is on the LEFT (--shift-x N) they move right by N
// so they stay registered with the page they belong to.
//
// Usage: node add_cut_layer.mjs <in.psd> <layer.png> <out.psd> --name NAME --above LAYER
//             --width W --shift-x N [--hidden]
//   --width    new canvas width (default: the layer's width, if wider than the canvas)
//   --shift-x  columns added on the left (0 when the strip is on the right)
import * as fs from 'fs';
import * as path from 'path';
import { readPsd, writePsdBuffer, initializeCanvas } from 'ag-psd';
import { loadImage, createCanvas, createImageData } from 'canvas';

initializeCanvas(createCanvas, createImageData);
const args = process.argv.slice(2);
const pos = [];
const opts = {};
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--hidden') opts.hidden = true;
  else if (args[i].startsWith('--')) opts[args[i].slice(2)] = args[++i];
  else pos.push(args[i]);
}
const [inPath, pngPath, outPath] = pos;
if (!outPath || !opts.above || !opts.name) {
  console.error('Usage: node add_cut_layer.mjs <in.psd> <layer.png> <out.psd> --name NAME --above LAYER [--width W] [--shift-x N] [--hidden]');
  process.exit(1);
}

/** Moves a layer (and its children) `dx` columns right. Text layers keep their box: left/right and the text transform move. */
function shiftLayer(layer, dx) {
  if (!dx) return;
  if (layer.children) layer.children.forEach((c) => shiftLayer(c, dx));
  if (layer.left !== undefined && layer.right !== undefined) { layer.left += dx; layer.right += dx; }
  if (layer.text?.transform) layer.text.transform[4] += dx;
  if (layer.mask && layer.mask.left !== undefined) { layer.mask.left += dx; layer.mask.right += dx; }
}

const psd = readPsd(fs.readFileSync(inPath), { useImageData: true, skipCompositeImageData: true, skipThumbnail: true });
const at = psd.children.findIndex((l) => l.name === opts.above);
if (at === -1) {
  console.error(`no top-level layer named "${opts.above}" in ${inPath} (has: ${psd.children.map((l) => l.name).join(', ')})`);
  process.exit(1);
}
const img = await loadImage(pngPath);
const canvas = createCanvas(img.width, img.height);
canvas.getContext('2d').drawImage(img, 0, 0);
const d = canvas.getContext('2d').getImageData(0, 0, img.width, img.height);
const shiftX = Number(opts['shift-x'] || 0);
const newWidth = Number(opts.width || Math.max(psd.width, img.width));
if (newWidth < psd.width + shiftX) {
  console.error(`--width ${newWidth} is smaller than the canvas ${psd.width} + shift ${shiftX}`);
  process.exit(1);
}
if (img.height !== psd.height) {
  console.error(`layer height ${img.height} differs from the canvas height ${psd.height}`);
  process.exit(1);
}
psd.children.forEach((l) => shiftLayer(l, shiftX));
psd.width = newWidth;
const data = new Uint8Array(d.data.buffer, d.data.byteOffset, d.data.byteLength);
psd.children.splice(at + 1, 0, {
  name: opts.name, hidden: !!opts.hidden, left: 0, top: 0, right: img.width, bottom: img.height,
  imageData: { data, width: img.width, height: img.height },
});
// the merged preview: the new layer covers the whole canvas
psd.imageData = { data, width: img.width, height: img.height };
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, writePsdBuffer(psd, { generateThumbnail: false }));
console.log(`${path.basename(outPath)}: "${opts.name}" ${img.width}x${img.height} above "${opts.above}", canvas ${newWidth}x${psd.height}, shift ${shiftX}`);
