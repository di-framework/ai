import { describe, expect, test } from 'bun:test';
import { l2Normalize } from '../src/embedding/fake-embedding-model.ts';
import { assignLevel, cosine, HnswIndex } from '../src/vectorstore/adapters/hnsw.ts';

describe('HnswIndex', () => {
  test('returns nearest neighbors and restores from a snapshot', () => {
    const index = new HnswIndex({ M: 8, efConstruction: 32, efSearch: 32 });
    index.insert('a', [1, 0]);
    index.insert('b', [0.9, 0.1]);
    index.insert('c', [0, 1]);
    const hits = index.search([1, 0], 2);
    expect(hits[0]?.id).toBe('a');
    expect(hits.map((hit) => hit.id)).toContain('b');

    const restored = HnswIndex.restore(
      index.snapshot(),
      new Map([
        ['a', Float32Array.from([1, 0])],
        ['b', Float32Array.from([0.9, 0.1])],
        ['c', Float32Array.from([0, 1])],
      ]),
    );
    expect(restored.search([0, 1], 1)[0]?.id).toBe('c');
    restored.remove('c');
    expect(restored.search([0, 1], 1)[0]?.id).not.toBe('c');
  });

  test('recall@10 against brute force is at least 99% on random unit vectors', () => {
    const dim = 32;
    const count = 800;
    const k = 10;
    const rng = mulberry32(7);
    const vectors = Array.from({ length: count }, (_, index) => ({
      id: `v${index}`,
      embedding: randomUnit(dim, rng),
    }));
    const index = new HnswIndex({ M: 16, efConstruction: 200, efSearch: 64 });
    for (const vector of vectors) index.insert(vector.id, vector.embedding);

    let hits = 0;
    let total = 0;
    for (let query = 0; query < 40; query++) {
      const q = randomUnit(dim, rng);
      const exact = vectors
        .map((vector) => ({ id: vector.id, score: cosine(q, vector.embedding) }))
        .sort((left, right) => right.score - left.score || left.id.localeCompare(right.id))
        .slice(0, k)
        .map((item) => item.id);
      const ann = new Set(index.search(q, k).map((item) => item.id));
      total += exact.length;
      hits += exact.filter((id) => ann.has(id)).length;
    }
    expect(hits / total).toBeGreaterThanOrEqual(0.99);
  }, 20000);

  test('replaces existing ids, ignores empty queries, and restores partial snapshots', () => {
    const index = new HnswIndex({ M: 4, efConstruction: 16, efSearch: 8 });
    expect(index.search([1, 0], 1)).toEqual([]);
    expect(index.search([1, 0], 0)).toEqual([]);
    index.insert('keep', [1, 0]);
    index.insert('keep', [0.99, 0.1]);
    index.insert('drop', [0, 1]);
    index.remove('drop');
    index.remove('missing');
    expect(index.search([0, 1], 1)[0]?.id).toBe('keep');
    expect(index.size).toBe(1);

    const high = highLevelId(index.mL);
    index.insert(high, [0.2, 0.8]);
    expect(index.search([0.2, 0.8], 1)[0]?.id).toBe(high);

    (index as unknown as { entryPoint: string | null }).entryPoint = 'ghost';
    index.insert('recovered', [0.7, 0.7]);
    expect(index.search([0.7, 0.7], 1)[0]?.id).toBe('recovered');
    (index as unknown as { entryPoint: string | null }).entryPoint = 'ghost';
    expect(index.search([1, 0], 1)).toEqual([]);

    const snapshot = index.snapshot();
    const restored = HnswIndex.restore(
      { ...snapshot, entryPoint: 'missing' },
      new Map([['recovered', Float32Array.from([0.7, 0.7])]]),
    );
    expect(restored.search([0.7, 0.7], 1)[0]?.id).toBe('recovered');
    expect(cosine([1, 0], [1, 0])).toBeCloseTo(1);
    expect(cosine([0, 0], [1, 0])).toBe(0);
    expect(() => cosine([1, 0], [1])).toThrow(/dimension mismatch/);
    expect(() => new HnswIndex({ M: 0 })).toThrow(/positive integer/);
  });

  test('selects diverse neighbors when several candidates cluster together', () => {
    const index = new HnswIndex({ M: 2, efConstruction: 32, efSearch: 16 });
    index.insert('origin', [1, 0]);
    index.insert('near', [0.98, 0.1]);
    index.insert('also-near', [0.97, 0.12]);
    index.insert('far', [0, 1]);
    const hits = index.search([1, 0], 3);
    expect(hits[0]?.id).toBe('origin');
    expect(hits.map((hit) => hit.id)).toEqual(expect.arrayContaining(['origin', 'near']));
  });
});

function highLevelId(mL: number): string {
  for (let index = 0; index < 50_000; index++) {
    const id = `lvl-${index}`;
    if (assignLevel(id, mL) >= 2) return id;
  }
  throw new Error('did not find a high-level id');
}

function randomUnit(dimensions: number, rng: () => number): number[] {
  return l2Normalize(Array.from({ length: dimensions }, () => rng() * 2 - 1));
}

function mulberry32(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (value + 0x6d2b79f5) >>> 0;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
