// `wasm-similarity` is mocked out to an empty module in
// `tests/preload-wasm-mock.ts` (registered via `bunfig.toml`'s
// `[test].preload`), which runs before every test file loads. `bun-sqlite.ts`
// memoizes its wasm import for the lifetime of the process, and `mock.module`
// cannot override an already-resolved real import, so the preload step is
// load-bearing for deterministically exercising the pure-JS `cosine`
// fallback below (bun's per-file test order is not alphabetical/stable).
import { Database } from 'bun:sqlite';
import { expect, test } from 'bun:test';

const { textDocument } = await import('../src/document/document.ts');
const { FakeEmbeddingModel } = await import('../src/embedding/fake-embedding-model.ts');
const { BunSqliteVectorStore } = await import('../src/vectorstore/adapters/bun-sqlite.ts');
const { filterExpression, filterKey, filterValue } = await import(
  '../src/vectorstore/filter/index.ts'
);

test('BunSqliteVectorStore falls back to JS cosine similarity when wasm-similarity is unusable', async () => {
  const db = new Database(':memory:');
  const store = new BunSqliteVectorStore({
    db,
    embeddingModel: new FakeEmbeddingModel({ dimensions: 8 }),
  });
  await store.add([
    textDocument('hello world', { kind: 'greeting' }, 'a'),
    textDocument('goodbye friend', {}, 'b'),
  ]);
  const hits = await store.similaritySearchQuery('hello');
  expect(hits[0]?.id).toBe('a');
});

test('BunSqliteVectorStore.add rejects embeddings with mismatched dimensions', async () => {
  const db = new Database(':memory:');
  const badModel = {
    dimensions: 8,
    embed: () => [1, 2, 3],
    embedDocument: () => [1, 2, 3],
  };
  const store = new BunSqliteVectorStore({ db, embeddingModel: badModel });
  await expect(store.add([textDocument('x', {}, 'a')])).rejects.toThrow(
    'Embedding dimension mismatch',
  );
});

test('BunSqliteVectorStore.delete and deleteByFilter remove rows', async () => {
  const db = new Database(':memory:');
  const store = new BunSqliteVectorStore({
    db,
    embeddingModel: new FakeEmbeddingModel({ dimensions: 8 }),
  });
  await store.add([
    textDocument('alpha', { group: 'x' }, 'a'),
    textDocument('beta', { group: 'y' }, 'b'),
    textDocument('gamma', { group: 'x' }, 'c'),
  ]);

  await store.delete(['b']);
  let hits = await store.similaritySearch({
    query: 'alpha',
    topK: 10,
    similarityThreshold: 0,
    filterExpression: null,
  });
  expect(hits.map((h) => h.id).sort()).toEqual(['a', 'c']);

  await store.deleteByFilter(filterExpression('EQ', filterKey('group'), filterValue('x')));
  hits = await store.similaritySearch({
    query: 'alpha',
    topK: 10,
    similarityThreshold: 0,
    filterExpression: null,
  });
  expect(hits).toHaveLength(0);
});

test('BunSqliteVectorStore.similaritySearch applies filterExpression', async () => {
  const db = new Database(':memory:');
  const store = new BunSqliteVectorStore({
    db,
    embeddingModel: new FakeEmbeddingModel({ dimensions: 8 }),
  });
  await store.add([
    textDocument('alpha content', { group: 'x' }, 'a'),
    textDocument('alpha content too', { group: 'y' }, 'b'),
  ]);

  const hits = await store.similaritySearch({
    query: 'alpha',
    topK: 10,
    similarityThreshold: 0,
    filterExpression: filterExpression('EQ', filterKey('group'), filterValue('y')),
  });
  expect(hits.map((h) => h.id)).toEqual(['b']);
});
