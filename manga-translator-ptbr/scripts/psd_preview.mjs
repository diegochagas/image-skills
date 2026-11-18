import { readPsd } from 'ag-psd';
import { createCanvas, createImageData } from 'canvas';

// The merged preview: the topmost full raster (the lettered / cleaned art)
// on white. A second, decoded read of the file - the raw read used for the
// byte-exact rewrite has no pixels to compose from.
export function compositeFromTopRaster(buf, W, H) {
  try {
    const q = readPsd(buf, { useImageData: true, skipCompositeImageData: true, skipThumbnail: true });
    const t = (q.children || []).filter((l) => !l.text && l.imageData && l.imageData.width > 0).pop();
    if (!t) return undefined;
    const c = createCanvas(W, H); const ctx = c.getContext('2d');
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
    ctx.putImageData(createImageData(new Uint8ClampedArray(t.imageData.data.buffer, t.imageData.data.byteOffset, t.imageData.data.byteLength), t.imageData.width, t.imageData.height), t.left || 0, t.top || 0);
    const d = ctx.getImageData(0, 0, W, H);
    return { data: new Uint8Array(d.data.buffer, d.data.byteOffset, d.data.byteLength), width: W, height: H };
  } catch (e) { return undefined; }
}
