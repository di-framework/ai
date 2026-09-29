#!/usr/bin/env bun
/**
 * Generate the English dictionary and fetch the pinned PP-OCRv4 rec ONNX.
 * See examples/ocr/SOURCE.md.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DEST_DIR = join(ROOT, 'examples/ocr/models');
const DEST = join(DEST_DIR, 'en_PP-OCRv4_rec_mobile.onnx');
const SHA256 = 'e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318';
const URLS = [
  'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.onnx',
  'https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/en_PP-OCRv4_rec_infer.onnx',
];

function sha256(buf: Uint8Array): string {
  return createHash('sha256').update(buf).digest('hex');
}

/** Model class order: ASCII 48–126, ASCII 33–47, then space. The decoder adds CTC blank. */
export function ensureOcrDictionary(directory = join(ROOT, 'examples/ocr')): string {
  const path = join(directory, 'en_dict.txt');
  const chars = [
    ...Array.from({ length: 79 }, (_, i) => String.fromCharCode(48 + i)),
    ...Array.from({ length: 15 }, (_, i) => String.fromCharCode(33 + i)),
    ' ',
  ];
  const text = `${chars.join('\n')}\n`;
  if (existsSync(path)) {
    if (readFileSync(path, 'utf8') !== text)
      throw new Error("examples/ocr dictionary does not match the pinned model's character order");
    return path;
  }
  mkdirSync(directory, { recursive: true });
  writeFileSync(path, text);
  return path;
}

export async function ensureOcrModel(): Promise<string> {
  ensureOcrDictionary();
  mkdirSync(DEST_DIR, { recursive: true });
  if (existsSync(DEST)) {
    const got = sha256(readFileSync(DEST));
    if (got === SHA256) return DEST;
    throw new Error(`examples/ocr model sha256 mismatch: ${got} (expected ${SHA256})`);
  }
  let last = '';
  for (const url of URLS) {
    try {
      const res = await fetch(url);
      if (!res.ok) {
        last = `${url} HTTP ${res.status}`;
        continue;
      }
      const buf = new Uint8Array(await res.arrayBuffer());
      const got = sha256(buf);
      if (got !== SHA256) {
        last = `${url} sha256 ${got}`;
        continue;
      }
      writeFileSync(DEST, buf);
      return DEST;
    } catch (err) {
      last = `${url}: ${err}`;
    }
  }
  throw new Error(`failed to fetch OCR model: ${last}`);
}

if (import.meta.main) {
  const path = await ensureOcrModel();
  console.log(`ok ${path}`);
}
