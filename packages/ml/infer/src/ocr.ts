import type { Session, Tensor } from './session.ts';

/** PP-OCR rec line height. Width is padded. */
export const REC_HEIGHT = 48;
export const REC_WIDTH = 320;

export type RgbImage = {
  width: number;
  height: number;
  /** Packed RGB, length width*height*3. */
  data: Uint8Array;
};

/**
 * Resize a line crop to height 48, pad to `width`, NCHW float32 in [-1, 1].
 * Channels are BGR (Paddle rec).
 */
export function preprocessLine(image: RgbImage, width = REC_WIDTH, height = REC_HEIGHT): Tensor {
  if (image.width < 1 || image.height < 1) {
    throw new Error('empty image');
  }
  const ratio = image.width / image.height;
  const resizedW = Math.max(1, Math.min(width, Math.ceil(height * ratio)));
  const chw = new Float32Array(3 * height * width);
  for (let y = 0; y < height; y++) {
    const sy = Math.min(image.height - 1, Math.floor(((y + 0.5) * image.height) / height));
    for (let x = 0; x < resizedW; x++) {
      const sx = Math.min(image.width - 1, Math.floor(((x + 0.5) * image.width) / resizedW));
      const si = (sy * image.width + sx) * 3;
      const r = image.data[si] ?? 0;
      const g = image.data[si + 1] ?? 0;
      const b = image.data[si + 2] ?? 0;
      const dst = y * width + x;
      chw[0 * height * width + dst] = (b / 255 - 0.5) / 0.5;
      chw[1 * height * width + dst] = (g / 255 - 0.5) / 0.5;
      chw[2 * height * width + dst] = (r / 255 - 0.5) / 0.5;
    }
  }
  return { data: chw, dims: [1, 3, height, width] };
}

/** Greedy CTC: blank is index 0. Extra classes beyond charset are skipped. */
export function ctcDecode(logits: Tensor, charset: string[]): { text: string; confidence: number } {
  const dims = logits.dims;
  const tLen = dims.length === 3 ? dims[1] : dims[0];
  const nClass = dims.length === 3 ? dims[2] : dims[1];
  if (tLen === undefined || nClass === undefined)
    throw new Error('OCR logits must have shape [T,C] or [1,T,C]');
  const data = logits.data;
  const chars: string[] = [];
  let last = 0;
  let confSum = 0;
  let confN = 0;
  for (let t = 0; t < tLen; t++) {
    let best = 0;
    let bestV = -Infinity;
    const row = t * nClass;
    for (let c = 0; c < nClass; c++) {
      const v = data[row + c] ?? -Infinity;
      if (v > bestV) {
        bestV = v;
        best = c;
      }
    }
    if (best !== 0 && best !== last) {
      const ch = charset[best];
      if (ch !== undefined && ch.length > 0) {
        chars.push(ch);
        confSum += bestV;
        confN += 1;
      }
    }
    last = best;
  }
  return {
    text: chars.join(''),
    confidence: confN === 0 ? 0 : confSum / confN,
  };
}

export function loadCharset(text: string): string[] {
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  return ['', ...lines];
}

export async function recognizeLine(
  session: Session,
  image: RgbImage,
  charset: string[],
): Promise<{ text: string; confidence: number }> {
  const inputName = session.inputs[0];
  if (!inputName) {
    throw new Error('OCR session has no inputs');
  }
  const feed = preprocessLine(image);
  const out = await session.run({ [inputName]: feed });
  const name = session.outputs[0];
  if (!name || !out[name]) {
    throw new Error('OCR session has no outputs');
  }
  const logits = out[name];
  if (!(logits.data instanceof Float32Array)) throw new Error('OCR output must be float32');
  return ctcDecode({ data: logits.data, dims: logits.dims }, charset);
}

const FONT_5X7: Record<string, number[]> = {
  H: [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
  E: [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
  L: [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
  O: [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
  ' ': [0, 0, 0, 0, 0, 0, 0],
};

/** White background, black 5×7 glyphs. Used as the OCR fixture. */
export function renderGlyphLine(text: string, scale = 6, pad = 8): RgbImage {
  const blank = FONT_5X7[' '];
  if (!blank) throw new Error('missing space glyph');
  const glyphs = [...text].map((ch) => FONT_5X7[ch] ?? blank);
  const gw = 5;
  const gh = 7;
  const gap = 1;
  const innerW = glyphs.length * (gw + gap) - gap;
  const width = pad * 2 + innerW * scale;
  const height = pad * 2 + gh * scale;
  const data = new Uint8Array(width * height * 3);
  data.fill(255);
  for (let gi = 0; gi < glyphs.length; gi++) {
    const g = glyphs[gi];
    if (!g) continue;
    const ox = pad + gi * (gw + gap) * scale;
    for (let row = 0; row < gh; row++) {
      const bits = g[row] ?? 0;
      for (let col = 0; col < gw; col++) {
        if ((bits & (1 << (gw - 1 - col))) === 0) continue;
        for (let dy = 0; dy < scale; dy++) {
          for (let dx = 0; dx < scale; dx++) {
            const x = ox + col * scale + dx;
            const y = pad + row * scale + dy;
            const i = (y * width + x) * 3;
            data[i] = 0;
            data[i + 1] = 0;
            data[i + 2] = 0;
          }
        }
      }
    }
  }
  return { width, height, data };
}
