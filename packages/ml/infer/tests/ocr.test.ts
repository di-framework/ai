import { describe, expect, test } from 'bun:test';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ensureOcrDictionary, ensureOcrModel } from '../../scripts/fetch-ocr-model.ts';
import {
  ctcDecode,
  loadCharset,
  preprocessLine,
  recognizeLine,
  renderGlyphLine,
} from '../src/ocr.ts';
import { Session } from '../src/session.ts';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const dictPath = join(root, 'examples', 'ocr', 'en_dict.txt');

describe('OCR rec', () => {
  test("generates the OCR dictionary with the model's character order", async () => {
    const directory = await mkdtemp(join(tmpdir(), 'di-ml-ocr-dict-'));
    try {
      const path = ensureOcrDictionary(directory);
      const chars = loadCharset(await readFile(path, 'utf8'));
      expect(chars).toHaveLength(96); // 95 printable ASCII characters plus CTC blank.
      expect(new Set(chars).size).toBe(96);
      expect([chars[0], chars[1], chars[18], chars[79], chars[80], chars[94], chars[95]]).toEqual([
        '',
        '0',
        'A',
        '~',
        '!',
        '/',
        ' ',
      ]);
      expect(ensureOcrDictionary(directory)).toBe(path);
      await writeFile(path, 'wrong character order');
      expect(() => ensureOcrDictionary(directory)).toThrow('character order');
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test('rejects an empty crop and a session with no tensors', async () => {
    expect(() => preprocessLine({ width: 0, height: 48, data: new Uint8Array() })).toThrow(
      'empty image',
    );
    const image = renderGlyphLine('H');
    const noInputs = {
      inputs: [],
      outputs: ['y'],
      run: async () => ({}),
    } as unknown as Session;
    await expect(recognizeLine(noInputs, image, [''])).rejects.toThrow('no inputs');
    const noOutputs = {
      inputs: ['x'],
      outputs: ['y'],
      run: async () => ({}),
    } as unknown as Session;
    await expect(recognizeLine(noOutputs, image, [''])).rejects.toThrow('no outputs');
  });

  test('ctcDecode skips blanks and repeats', () => {
    const charset = ['', 'H', 'E', 'L', 'O'];
    const t = 6;
    const c = 5;
    const data = new Float32Array(t * c);
    const picks = [1, 1, 0, 2, 3, 4];
    for (let i = 0; i < t; i++) {
      const pick = picks[i];
      if (pick === undefined) throw new Error('missing class index');
      data[i * c + pick] = 1;
    }
    const { text } = ctcDecode({ data, dims: [1, t, c] }, charset);
    expect(text).toBe('HELO');
  });

  test('reads HELLO from a rendered line', async () => {
    const model = await ensureOcrModel();
    const session = await Session.fromPath(model);
    expect(session.inputs).toEqual(['x']);
    expect(session.outputs.length).toBe(1);

    const charset = loadCharset(await readFile(dictPath, 'utf8'));
    const image = renderGlyphLine('HELLO', 8, 12);
    const { text, confidence } = await recognizeLine(session, image, charset);
    expect(text).toBe('HELLO');
    expect(confidence).toBeGreaterThan(0.5);
    await session.release();
  }, 120_000); // A clean checkout downloads and checksums the OCR model first.
});
