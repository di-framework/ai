import { Database } from 'bun:sqlite';
import { expect, test } from 'bun:test';
import { textDocument } from '../src/document/document.ts';
import { FakeEmbeddingModel } from '../src/embedding/fake-embedding-model.ts';
import { BunSqliteVectorStore } from '../src/vectorstore/adapters/bun-sqlite.ts';

test('BunSqliteVectorStore persists and searches documents', async () => {
  const db = new Database(':memory:');
  const store = new BunSqliteVectorStore({
    db,
    embeddingModel: new FakeEmbeddingModel({ dimensions: 8 }),
  });
  await store.add([
    textDocument('hello world', { kind: 'greeting' }, 'a'),
    textDocument('goodbye', {}, 'b'),
  ]);
  const hits = await store.similaritySearchQuery('hello');
  expect(hits[0]?.id).toBe('a');
});
