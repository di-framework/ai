import { beforeAll, describe, expect, test } from 'bun:test';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Session } from '../src/index.ts';
import { generateInferenceFixtures } from './fixtures.ts';

const xorOnnx = join(dirname(fileURLToPath(import.meta.url)), '..', 'testdata', 'xor.onnx');
beforeAll(() => generateInferenceFixtures(join(import.meta.dir, '../testdata')));

describe('Session', () => {
  test('integer feeds gather embeddings and preserve integer outputs exactly', async () => {
    const session = await Session.fromPath(join(import.meta.dir, '../testdata/integer.onnx'));
    try {
      const out = await session.run({
        ids: { data: new BigInt64Array([2n, 0n]), dims: [2] },
        mask: { data: new Int32Array([1, 0]), dims: [2] },
        large: { data: new BigInt64Array([9007199254740993n]), dims: [1] },
      });
      expect(out.embedding?.data).toEqual(new Float32Array([5, 6, 1, 2]));
      expect(out.embedding?.dims).toEqual([2, 2]);
      expect(out.mask_out?.data).toEqual(new Int32Array([1, 0]));
      expect(out.large_out?.data).toEqual(new BigInt64Array([9007199254740993n]));
    } finally {
      await session.release();
    }
    await session.release();
    await expect(session.run({})).rejects.toThrow('released');
  });

  test('loads XOR I/O names and returns a [4,1] output', async () => {
    const session = await Session.fromPath(xorOnnx);
    expect(session.inputs).toEqual(['input']);
    expect(session.outputs).toEqual(['output']);

    const input = new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]);
    const out = await session.run({
      input: { data: input, dims: [4, 2] },
    });
    const output = out.output;
    if (!output) throw new Error('missing output');
    expect(output.dims).toEqual([4, 1]);
    expect(output.data.length).toBe(4);
    expect([...output.data].map((v) => (Number(v) > 0.5 ? 1 : 0))).toEqual([0, 1, 1, 0]);
    for (const v of out.output?.data ?? []) {
      expect(Number.isFinite(v)).toBe(true);
    }
    await session.release();
  });

  test('fromBytes matches fromPath', async () => {
    const bytes = new Uint8Array(await Bun.file(xorOnnx).arrayBuffer());
    const session = await Session.fromBytes(bytes);
    const out = await session.run({
      input: { data: new Float32Array([0, 1]), dims: [1, 2] },
    });
    expect(out.output?.dims).toEqual([1, 1]);
    expect(Number.isFinite(out.output?.data[0])).toBe(true);
    await session.release();
  });
});
