import { describe, expect, test } from 'bun:test';
import { loadWasmSimilarity } from '../src/vectorstore/adapters/wasm-similarity-loader.ts';

describe('loadWasmSimilarity', () => {
  test('returns null when the module has no cosine ranker', async () => {
    expect(
      await loadWasmSimilarity(
        new URL('./fixtures/empty-wasm-similarity.ts', import.meta.url).href,
      ),
    ).toBeNull();
  });

  test('returns null when the specifier cannot be imported', async () => {
    expect(await loadWasmSimilarity('this-wasm-module-does-not-exist')).toBeNull();
  });

  test('returns a module that exposes cosine_similarity_dataspace', async () => {
    const wasm = await loadWasmSimilarity(
      new URL('./fixtures/fake-wasm-similarity.ts', import.meta.url).href,
    );
    expect(typeof wasm?.cosine_similarity_dataspace).toBe('function');
    expect(
      wasm?.cosine_similarity_dataspace(new Float64Array([1, 0]), 1, 2, new Float64Array([1, 0])),
    ).toEqual([1, 0]);
  });
});
