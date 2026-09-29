import { Database } from 'bun:sqlite';
import { describe, expect, test } from 'bun:test';
import { document, textDocument } from '../src/document/document.ts';
import { FakeEmbeddingModel } from '../src/embedding/fake-embedding-model.ts';
import { PrecomputedEmbeddingModel } from '../src/embedding/precomputed-embedding-model.ts';
import { BunSqliteVectorStore, rankCosineScores } from '../src/vectorstore/adapters/bun-sqlite.ts';
import { searchRequest } from '../src/vectorstore/search-request.ts';

describe('BunSqliteVectorStore ANN', () => {
  test('uses Document.embedding and SearchRequest.queryEmbedding', async () => {
    const db = new Database(':memory:');
    const store = new BunSqliteVectorStore({
      db,
      embeddingModel: new PrecomputedEmbeddingModel(2),
    });
    await store.add([
      document({ id: 'east', text: 'east', embedding: [1, 0] }),
      document({ id: 'north', text: 'north', embedding: [0, 1] }),
    ]);
    const hits = await store.similaritySearch(
      searchRequest({ query: 'ignored', queryEmbedding: [1, 0], topK: 1 }),
    );
    expect(hits[0]?.id).toBe('east');
  });

  test('auto mode exact-scans below the limit and uses HNSW above it', async () => {
    const db = new Database(':memory:');
    const store = new BunSqliteVectorStore({
      db,
      embeddingModel: new PrecomputedEmbeddingModel(2),
      exactScanLimit: 3,
    });
    await store.add([
      document({ id: 'a', text: 'a', embedding: [1, 0] }),
      document({ id: 'b', text: 'b', embedding: [0.9, 0.1] }),
    ]);
    expect(
      (await store.similaritySearch(searchRequest({ queryEmbedding: [1, 0], topK: 1 })))[0]?.id,
    ).toBe('a');
    await store.add([
      document({ id: 'c', text: 'c', embedding: [0.2, 0.8] }),
      document({ id: 'd', text: 'd', embedding: [0, 1] }),
    ]);
    const hits = await store.similaritySearch(searchRequest({ queryEmbedding: [0, 1], topK: 1 }));
    expect(hits[0]?.id).toBe('d');
  });

  test('ann and exact modes agree on the top hit', async () => {
    const db = new Database(':memory:');
    const model = new FakeEmbeddingModel({ dimensions: 8 });
    const ann = new BunSqliteVectorStore({ db, embeddingModel: model, searchMode: 'ann' });
    await ann.add([
      textDocument('alpha document about cats', {}, 'a'),
      textDocument('beta document about dogs', {}, 'b'),
      textDocument('gamma notes on feline care', {}, 'c'),
    ]);
    const exact = new BunSqliteVectorStore({ db, embeddingModel: model, searchMode: 'exact' });
    const query = searchRequest({ query: 'cats feline', topK: 2 });
    expect((await exact.similaritySearch(query))[0]?.id).toBe(
      (await ann.similaritySearch(query))[0]?.id,
    );
  });

  test('reads legacy JSON embeddings', async () => {
    const db = new Database(':memory:');
    db.run(
      `CREATE TABLE ai_vectors (id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata TEXT NOT NULL, embedding TEXT NOT NULL)`,
    );
    db.run(`INSERT INTO ai_vectors VALUES ('json', 'hello world', '{}', ?)`, [
      JSON.stringify(new FakeEmbeddingModel({ dimensions: 8 }).embed('hello world')),
    ]);
    const store = new BunSqliteVectorStore({
      db,
      embeddingModel: new FakeEmbeddingModel({ dimensions: 8 }),
    });
    const hits = await store.similaritySearchQuery('hello');
    expect(hits[0]?.id).toBe('json');
  });

  test('rejects unsupported format versions and corrupt embeddings', async () => {
    const db = new Database(':memory:');
    new BunSqliteVectorStore({
      db,
      embeddingModel: new FakeEmbeddingModel({ dimensions: 8 }),
    });
    db.run(
      `INSERT OR REPLACE INTO ai_vectors_store_meta (key, value) VALUES ('formatVersion', 'v9')`,
    );
    expect(
      () =>
        new BunSqliteVectorStore({
          db,
          embeddingModel: new FakeEmbeddingModel({ dimensions: 8 }),
        }),
    ).toThrow(/Unsupported vector store format/);

    const corrupt = new Database(':memory:');
    const live = new BunSqliteVectorStore({
      db: corrupt,
      embeddingModel: new FakeEmbeddingModel({ dimensions: 8 }),
    });
    await live.add([textDocument('ok', {}, 'ok')]);
    corrupt.run(`UPDATE ai_vectors SET embedding = ? WHERE id = 'ok'`, [new Uint8Array([1, 2, 3])]);
    const reread = new BunSqliteVectorStore({
      db: corrupt,
      embeddingModel: new FakeEmbeddingModel({ dimensions: 8 }),
    });
    await expect(reread.similaritySearchQuery('ok')).rejects.toThrow(/corrupt/);
  });

  test('searchMode ann fails closed when the graph is missing', async () => {
    const db = new Database(':memory:');
    const exact = new BunSqliteVectorStore({
      db,
      embeddingModel: new PrecomputedEmbeddingModel(2),
      searchMode: 'exact',
    });
    await exact.add([document({ id: 'a', text: 'a', embedding: [1, 0] })]);
    const ann = new BunSqliteVectorStore({
      db,
      embeddingModel: new PrecomputedEmbeddingModel(2),
      searchMode: 'ann',
    });
    await expect(
      ann.similaritySearch(searchRequest({ queryEmbedding: [1, 0], topK: 1 })),
    ).rejects.toThrow('ANN graph is not ready');
  });

  test('restores a persisted HNSW graph and rebuilds when the snapshot is empty', async () => {
    const db = new Database(':memory:');
    const first = new BunSqliteVectorStore({
      db,
      embeddingModel: new PrecomputedEmbeddingModel(2),
      exactScanLimit: 2,
    });
    await first.add([
      document({ id: 'a', text: 'a', embedding: [1, 0] }),
      document({ id: 'b', text: 'b', embedding: [0.9, 0.1] }),
      document({ id: 'c', text: 'c', embedding: [0, 1] }),
    ]);
    const restored = new BunSqliteVectorStore({
      db,
      embeddingModel: new PrecomputedEmbeddingModel(2),
      exactScanLimit: 2,
    });
    expect(
      (await restored.similaritySearch(searchRequest({ queryEmbedding: [0, 1], topK: 1 })))[0]?.id,
    ).toBe('c');
    expect(await restored.get('a')).toMatchObject({ id: 'a', text: 'a' });
    expect(await restored.get('missing')).toBeNull();

    db.run(`DELETE FROM ai_vectors_hnsw`);
    const rebuilt = new BunSqliteVectorStore({
      db,
      embeddingModel: new PrecomputedEmbeddingModel(2),
      exactScanLimit: 2,
    });
    expect(
      (await rebuilt.similaritySearch(searchRequest({ queryEmbedding: [1, 0], topK: 1 })))[0]?.id,
    ).toBe('a');
  });

  test('rejects corrupt graph snapshots and invalid construction options', async () => {
    expect(
      () =>
        new BunSqliteVectorStore({
          db: new Database(':memory:'),
          embeddingModel: new PrecomputedEmbeddingModel(2),
          exactScanLimit: 0,
        }),
    ).toThrow(/exactScanLimit/);

    const db = new Database(':memory:');
    const store = new BunSqliteVectorStore({
      db,
      embeddingModel: new PrecomputedEmbeddingModel(2),
      searchMode: 'ann',
    });
    await store.add([document({ id: 'a', text: 'a', embedding: [1, 0] })]);
    db.run(`UPDATE ai_vectors_hnsw SET neighbors = 'not-json'`);
    const reread = new BunSqliteVectorStore({
      db,
      embeddingModel: new PrecomputedEmbeddingModel(2),
      searchMode: 'ann',
    });
    await expect(
      reread.similaritySearch(searchRequest({ queryEmbedding: [1, 0], topK: 1 })),
    ).rejects.toThrow(/corrupt/);

    const notArray = new Database(':memory:');
    const live = new BunSqliteVectorStore({
      db: notArray,
      embeddingModel: new PrecomputedEmbeddingModel(2),
      searchMode: 'ann',
    });
    await live.add([document({ id: 'a', text: 'a', embedding: [1, 0] })]);
    notArray.run(`UPDATE ai_vectors_hnsw SET neighbors = '1'`);
    const broken = new BunSqliteVectorStore({
      db: notArray,
      embeddingModel: new PrecomputedEmbeddingModel(2),
      searchMode: 'ann',
    });
    await expect(
      broken.similaritySearch(searchRequest({ queryEmbedding: [1, 0], topK: 1 })),
    ).rejects.toThrow(/corrupt/);
  });

  test('filters ANN candidates, accepts format v1, and decodes alternate embedding payloads', async () => {
    const store = new BunSqliteVectorStore({
      db: new Database(':memory:'),
      embeddingModel: new PrecomputedEmbeddingModel(2),
      exactScanLimit: 1,
    });
    await store.add([
      document({ id: 'keep', text: 'keep', embedding: [1, 0], metadata: { group: 'yes' } }),
      document({ id: 'drop', text: 'drop', embedding: [0.99, 0.01], metadata: { group: 'no' } }),
    ]);
    const hits = await store.similaritySearch(
      searchRequest({
        queryEmbedding: [1, 0],
        topK: 1,
        filterExpression: 'group == "yes"',
      }),
    );
    expect(hits[0]?.id).toBe('keep');
    expect(
      await store.similaritySearch(searchRequest({ queryEmbedding: [1, 0], topK: 1 })),
    ).toHaveLength(1);

    const legacy = new Database(':memory:');
    const opened = new BunSqliteVectorStore({
      db: legacy,
      embeddingModel: new PrecomputedEmbeddingModel(2),
    });
    legacy.run(
      `INSERT OR REPLACE INTO ai_vectors_store_meta (key, value) VALUES ('formatVersion', '1')`,
    );
    const v1 = new BunSqliteVectorStore({
      db: legacy,
      embeddingModel: new PrecomputedEmbeddingModel(2),
    });
    await v1.add([document({ id: 'legacy', text: 'legacy', embedding: [0, 1] })]);
    expect(
      (await v1.similaritySearch(searchRequest({ queryEmbedding: [0, 1], topK: 1 })))[0]?.id,
    ).toBe('legacy');
    expect(opened.name).toBe('BunSqliteVectorStore');
  });

  test('decodes JSON, ArrayBuffer, and typed-array embeddings and rejects corrupt rows', async () => {
    const db = new ScriptedVectorDatabase();
    db.rows.push({
      id: 'json',
      text: 'json',
      metadata: '{}',
      embedding: JSON.stringify([1, 0]),
    });
    const store = new BunSqliteVectorStore({
      db,
      embeddingModel: new PrecomputedEmbeddingModel(2),
    });
    expect(
      (await store.similaritySearch(searchRequest({ queryEmbedding: [1, 0], topK: 1 })))[0]?.id,
    ).toBe('json');

    db.rows[0] = {
      id: 'bytes',
      text: 'bytes',
      metadata: '{}',
      embedding: new Float32Array([0, 1]).buffer,
    };
    db.cacheBust();
    expect(
      (await store.similaritySearch(searchRequest({ queryEmbedding: [0, 1], topK: 1 })))[0]?.id,
    ).toBe('bytes');

    db.rows[0] = {
      id: 'view',
      text: 'view',
      metadata: '{}',
      embedding: new Float32Array([1, 0]),
    };
    db.cacheBust();
    expect(
      (await store.similaritySearch(searchRequest({ queryEmbedding: [1, 0], topK: 1 })))[0]?.id,
    ).toBe('view');

    db.rows[0] = { id: 'bad-json', text: 'x', metadata: '{', embedding: new Uint8Array(8) };
    db.cacheBust();
    await expect(
      store.similaritySearch(searchRequest({ queryEmbedding: [1, 0], topK: 1 })),
    ).rejects.toThrow(/corrupt/);

    db.rows[0] = { id: 'bad-list', text: 'x', metadata: '{}', embedding: '1' };
    db.cacheBust();
    await expect(
      store.similaritySearch(searchRequest({ queryEmbedding: [1, 0], topK: 1 })),
    ).rejects.toThrow(/dimension mismatch/);

    db.rows[0] = { id: 'bad-json-emb', text: 'x', metadata: '{}', embedding: '{' };
    db.cacheBust();
    await expect(
      store.similaritySearch(searchRequest({ queryEmbedding: [1, 0], topK: 1 })),
    ).rejects.toThrow(/corrupt/);

    db.rows[0] = { id: 'odd', text: 'x', metadata: '{}', embedding: new Uint8Array([1, 2, 3]) };
    db.cacheBust();
    await expect(
      store.similaritySearch(searchRequest({ queryEmbedding: [1, 0], topK: 1 })),
    ).rejects.toThrow(/corrupt/);

    db.rows[0] = {
      id: 'nan',
      text: 'x',
      metadata: '{}',
      embedding: new Uint8Array(new Float32Array([Number.NaN, 0]).buffer),
    };
    db.cacheBust();
    await expect(
      store.similaritySearch(searchRequest({ queryEmbedding: [1, 0], topK: 1 })),
    ).rejects.toThrow(/corrupt/);

    db.rows[0] = { id: 'unknown', text: 'x', metadata: '{}', embedding: 12 };
    db.cacheBust();
    await expect(
      store.similaritySearch(searchRequest({ queryEmbedding: [1, 0], topK: 1 })),
    ).rejects.toThrow(/corrupt/);

    db.rows = [];
    db.cacheBust();
    expect(
      await store.similaritySearch(
        searchRequest({ queryEmbedding: [1, 0], topK: 1, filterExpression: 'group == "noop"' }),
      ),
    ).toEqual([]);
  });

  test('rankCosineScores uses wasm pairs and skips invalid indexes', () => {
    const rows = [
      { id: 'a', text: 'a', metadata: {}, embedding: [1, 0] },
      { id: 'b', text: 'b', metadata: {}, embedding: [0, 1] },
    ];
    const exact = rankCosineScores([1, 0], rows, null);
    expect(exact[0]?.r.id).toBe('a');
    const ranked = rankCosineScores([1, 0], rows, {
      cosine_similarity_dataspace: () => [0.1, 1, 0.9, 0, 0.2, 99],
    });
    expect(ranked.map((row) => row.r.id)).toEqual(['b', 'a']);
  });

  test('rejects non-finite precomputed embeddings and empty candidate sets', async () => {
    const store = new BunSqliteVectorStore({
      db: new Database(':memory:'),
      embeddingModel: new PrecomputedEmbeddingModel(2),
    });
    await expect(
      store.add([document({ id: 'bad', text: 'bad', embedding: [Number.POSITIVE_INFINITY, 0] })]),
    ).rejects.toThrow(/non-finite/);
    expect(
      await store.similaritySearch(searchRequest({ queryEmbedding: [1, 0], topK: 1 })),
    ).toEqual([]);
  });
});

class ScriptedVectorDatabase {
  readonly meta = new Map<string, string>();
  rows: Array<{ id: string; text: string; metadata: string; embedding: unknown }> = [];
  graph: Array<{ id: string; level: number; neighbors: string }> = [];
  generation = 0;

  run(sql: string, ...args: unknown[]): { changes?: number } {
    if (sql.startsWith('CREATE')) return {};
    if (sql.includes('_store_meta')) {
      this.meta.set(String(args[0]), String(args[1]));
      return {};
    }
    if (sql.includes('_hnsw') && sql.startsWith('INSERT')) {
      this.graph.push({ id: String(args[0]), level: Number(args[1]), neighbors: String(args[2]) });
      return {};
    }
    if (sql.includes('_hnsw') && sql.startsWith('DELETE')) {
      this.graph = [];
      return {};
    }
    if (sql.startsWith('INSERT') && sql.includes('ai_vectors')) {
      const row = {
        id: String(args[0]),
        text: String(args[1]),
        metadata: String(args[2]),
        embedding: args[3],
      };
      const index = this.rows.findIndex((item) => item.id === row.id);
      if (index >= 0) this.rows[index] = row;
      else this.rows.push(row);
      return {};
    }
    if (sql.startsWith('DELETE') && sql.includes('WHERE id')) {
      this.rows = this.rows.filter((row) => row.id !== String(args[0]));
    }
    return {};
  }

  query(sql: string): { all(...args: unknown[]): unknown[] } {
    return {
      all: (...args: unknown[]) => {
        if (sql.includes('_store_meta')) {
          const value = this.meta.get(String(args[0]));
          return value == null ? [] : [{ value }];
        }
        if (sql.includes('_hnsw')) return this.graph;
        if (sql.includes('WHERE id')) return this.rows.filter((row) => row.id === String(args[0]));
        return this.rows;
      },
    };
  }

  cacheBust(): void {
    this.generation += 1;
    this.meta.set('dataGeneration', String(this.generation));
  }
}
