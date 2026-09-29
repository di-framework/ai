import { describe, expect, test } from 'bun:test';
import { textDocument } from '../src/document/document.ts';
import { type EmbeddingModel, embedDocument } from '../src/embedding/embedding-model.ts';
import { FakeEmbeddingModel } from '../src/embedding/fake-embedding-model.ts';
import {
  emptyToolCallbackResolver,
  staticToolCallbackResolver,
} from '../src/model/tool/tool-callback-resolver.ts';
import { functionToolCallback } from '../src/tool/function-tool-callback.ts';
import { type PgClient, PgVectorStore } from '../src/vectorstore/adapters/pgvector.ts';
import {
  type VectorizeIndex,
  VectorizeVectorStore,
} from '../src/vectorstore/adapters/vectorize.ts';
import { filterExpression, filterKey, filterValue } from '../src/vectorstore/filter/index.ts';
import {
  SearchRequestBuilder,
  searchRequest,
  searchRequestBuilder,
} from '../src/vectorstore/search-request.ts';
import {
  similaritySearchQuery,
  type VectorStoreRetriever,
} from '../src/vectorstore/vector-store-retriever.ts';

describe('embedding-model.embedDocument', () => {
  test('delegates to model.embedDocument when present', async () => {
    const model = new FakeEmbeddingModel({ dimensions: 4 });
    const doc = textDocument('hello world', {}, 'd1');
    const result = await embedDocument(model, doc);
    expect(result).toEqual(model.embedDocument(doc));
  });

  test('falls back to model.embed(text) when embedDocument is not a function', async () => {
    const model: EmbeddingModel = {
      embed: (text: string) => [text.length, 0, 0],
    } as unknown as EmbeddingModel;
    const doc = textDocument('abcd', {}, 'd1');
    const result = await embedDocument(model, doc);
    expect(result).toEqual([4, 0, 0]);
  });

  test('falls back with empty string when doc.text is null', async () => {
    const model: EmbeddingModel = {
      embed: (text: string) => [text.length],
    } as unknown as EmbeddingModel;
    const doc = { id: 'x', text: null, media: null, metadata: {}, score: null };
    const result = await embedDocument(model, doc);
    expect(result).toEqual([0]);
  });
});

describe('tool-callback-resolver', () => {
  test('staticToolCallbackResolver resolves known tool names and undefined otherwise', () => {
    const cb = functionToolCallback({ name: 'echo', call: (input: unknown) => input });
    const resolver = staticToolCallbackResolver([cb]);
    expect(resolver.resolve('echo')).toBe(cb);
    expect(resolver.resolve('missing')).toBeUndefined();
  });

  test('emptyToolCallbackResolver always resolves undefined', () => {
    expect(emptyToolCallbackResolver.resolve('anything')).toBeUndefined();
  });
});

describe('vector-store-retriever.similaritySearchQuery', () => {
  test('delegates to retriever.similaritySearchQuery when implemented', async () => {
    const docs = [textDocument('a', {}, 'a')];
    const retriever: VectorStoreRetriever = {
      similaritySearch: async () => [],
      similaritySearchQuery: async () => docs,
    };
    const result = await similaritySearchQuery(retriever, 'q');
    expect(result).toBe(docs);
  });

  test('falls back to similaritySearch(searchRequest({query})) when not implemented', async () => {
    let capturedQuery: string | undefined;
    const docs = [textDocument('b', {}, 'b')];
    const retriever: VectorStoreRetriever = {
      similaritySearch: async (request) => {
        capturedQuery = request.query;
        return docs;
      },
    };
    const result = await similaritySearchQuery(retriever, 'hello');
    expect(result).toBe(docs);
    expect(capturedQuery).toBe('hello');
  });
});

describe('search-request', () => {
  test('searchRequest applies defaults and validates topK / threshold', () => {
    const req = searchRequest();
    expect(req).toEqual({ query: '', topK: 4, similarityThreshold: 0, filterExpression: null });

    expect(() => searchRequest({ topK: 0 })).toThrow('TopK should be positive.');
    expect(() => searchRequest({ similarityThreshold: -0.1 })).toThrow(
      'Similarity threshold must be in [0,1] range.',
    );
    expect(() => searchRequest({ similarityThreshold: 1.1 })).toThrow(
      'Similarity threshold must be in [0,1] range.',
    );
  });

  test('searchRequest parses a string filterExpression and accepts an object one', () => {
    const parsed = searchRequest({ filterExpression: 'group == "x"' });
    expect(parsed.filterExpression).not.toBeNull();

    const expr = filterExpression('EQ', filterKey('group'), filterValue('y'));
    const withObj = searchRequest({ filterExpression: expr });
    expect(withObj.filterExpression).toBe(expr);
  });

  test('SearchRequestBuilder builds a full request fluently', () => {
    const built = new SearchRequestBuilder()
      .query('hello')
      .topK(10)
      .similarityThreshold(0.5)
      .filterExpression(filterExpression('EQ', filterKey('a'), filterValue(1)))
      .build();
    expect(built.query).toBe('hello');
    expect(built.topK).toBe(10);
    expect(built.similarityThreshold).toBe(0.5);
    expect(built.filterExpression).not.toBeNull();

    expect(() => new SearchRequestBuilder().query(null as never)).toThrow('Query can not be null.');

    const acceptAll = searchRequestBuilder()
      .query('x')
      .similarityThreshold(0.9)
      .similarityThresholdAll()
      .filterExpression(undefined)
      .build();
    expect(acceptAll.similarityThreshold).toBe(0);
    expect(acceptAll.filterExpression).toBeNull();

    const withVector = searchRequestBuilder().query('q').queryEmbedding([1, 0]).build();
    expect(withVector.queryEmbedding).toEqual([1, 0]);
  });
});

describe('PgVectorStore', () => {
  function makeClient(): PgClient & { calls: { sql: string; params?: unknown[] }[] } {
    const rows: Array<{ id: string; content: string; metadata: string; score: number }> = [];
    const client = {
      calls: [] as { sql: string; params?: unknown[] }[],
      query<T = any>(sql: string, params?: unknown[]): { rows: T[] } {
        client.calls.push({ sql, params });
        if (sql.startsWith('INSERT')) {
          const [id, content, metadata] = params as [string, string, string, string];
          rows.push({ id, content, metadata, score: 1 });
          return { rows: [] };
        }
        if (sql.startsWith('DELETE')) {
          const [id] = params as [string];
          const idx = rows.findIndex((r) => r.id === id);
          if (idx >= 0) rows.splice(idx, 1);
          return { rows: [] };
        }
        // SELECT ... similarity search
        return { rows: rows.map((r) => ({ ...r, score: 0.9 })) as T[] };
      },
    };
    return client;
  }

  test('add/similaritySearch/delete round-trip through the pg client', async () => {
    const client = makeClient();
    const store = new PgVectorStore({ client, embeddingModel: new FakeEmbeddingModel() });
    expect(store.name).toBe('PgVectorStore');

    await store.add([textDocument('hello pgvector', { k: 'v' }, 'p1')]);
    expect(client.calls.some((c) => c.sql.startsWith('INSERT'))).toBe(true);

    const hits = await store.similaritySearch(searchRequest({ query: 'hello', topK: 5 }));
    expect(hits[0]?.id).toBe('p1');
    expect(hits[0]?.score).toBe(0.9);

    const viaQuery = await store.similaritySearchQuery('hello');
    expect(viaQuery[0]?.id).toBe('p1');

    await store.delete(['p1']);
    expect(client.calls.some((c) => c.sql.startsWith('DELETE'))).toBe(true);
  });

  test('filters out results below the similarity threshold and parses string metadata', async () => {
    const client: PgClient = {
      query: <T = any>(sql: string): { rows: T[] } => {
        if (sql.startsWith('SELECT')) {
          return {
            rows: [
              { id: 'low', content: 'low score', metadata: '{"a":1}', score: 0.1 },
              { id: 'high', content: 'high score', metadata: null, score: 0.95 },
            ] as T[],
          };
        }
        return { rows: [] };
      },
    };
    const store = new PgVectorStore({ client, embeddingModel: new FakeEmbeddingModel() });
    const hits = await store.similaritySearch(
      searchRequest({ query: 'x', similarityThreshold: 0.5 }),
    );
    expect(hits.map((h) => h.id)).toEqual(['high']);
    expect(hits[0]?.metadata).toEqual({});
  });

  test('uses a custom table name and name option', async () => {
    const client: PgClient = { query: () => ({ rows: [] }) };
    const store = new PgVectorStore({
      client,
      embeddingModel: new FakeEmbeddingModel(),
      table: 'custom_table',
      name: 'CustomStore',
    });
    expect(store.name).toBe('CustomStore');
    await store.add([textDocument('x', {}, 'x1')]);
  });

  test('get returns a document by id and honors queryEmbedding', async () => {
    const rows = [
      { id: 'p1', content: 'hello', metadata: '{"k":1}' },
      { id: 'p2', content: 'there', metadata: { k: 2 } },
    ];
    const client: PgClient = {
      query: <T = any>(sql: string, params?: unknown[]) => {
        if (sql.includes('WHERE id=')) {
          const row = rows.find((item) => item.id === params?.[0]);
          return { rows: (row ? [row] : []) as T[] };
        }
        if (sql.startsWith('SELECT')) {
          return { rows: [{ id: 'p1', content: 'hello', metadata: '{"k":1}', score: 0.8 }] as T[] };
        }
        return { rows: [] };
      },
    };
    const store = new PgVectorStore({ client, embeddingModel: new FakeEmbeddingModel() });
    expect(await store.get('p1')).toMatchObject({ id: 'p1', text: 'hello', metadata: { k: 1 } });
    expect(await store.get('p2')).toMatchObject({ metadata: { k: 2 } });
    expect(await store.get('missing')).toBeNull();
    const queryEmbedding = Array.from({ length: 64 }, (_, index) => (index === 0 ? 1 : 0));
    const hits = await store.similaritySearch(searchRequest({ queryEmbedding, topK: 1 }));
    expect(hits[0]?.id).toBe('p1');
  });
});

describe('VectorizeVectorStore', () => {
  function makeIndex(): VectorizeIndex & { upserted: unknown[] } {
    const index = {
      upserted: [] as unknown[],
      async upsert(vectors: unknown[]) {
        index.upserted.push(...vectors);
      },
      async query(_vector: number[], _options?: Record<string, unknown>) {
        return {
          matches: index.upserted.map((v, i) => ({
            id: (v as { id: string }).id,
            score: 1 - i * 0.1,
            metadata: (v as { metadata?: Record<string, unknown> }).metadata,
          })),
        };
      },
      async deleteByIds(ids: string[]) {
        index.upserted = index.upserted.filter((v) => !ids.includes((v as { id: string }).id));
      },
    };
    return index;
  }

  test('add/similaritySearch/similaritySearchQuery/delete round-trip', async () => {
    const index = makeIndex();
    const store = new VectorizeVectorStore({ index, embeddingModel: new FakeEmbeddingModel() });
    expect(store.name).toBe('VectorizeVectorStore');

    await store.add([textDocument('alpha', { g: 1 }, 'v1'), textDocument('beta', { g: 2 }, 'v2')]);
    expect(index.upserted).toHaveLength(2);

    const hits = await store.similaritySearch(searchRequest({ query: 'alpha', topK: 5 }));
    expect(hits.map((h) => h.id)).toContain('v1');

    const viaQuery = await store.similaritySearchQuery('alpha');
    expect(viaQuery.length).toBeGreaterThan(0);

    expect(await store.get('v1')).toMatchObject({ id: 'v1' });
    await store.delete(['v1']);
    expect(await store.get('v1')).toBeNull();
    const afterDelete = await store.similaritySearch(searchRequest({ query: 'alpha', topK: 5 }));
    expect(afterDelete.map((h) => h.id)).not.toContain('v1');
  });

  test('falls back to synthesized metadata/text when the doc cache misses and no matches', async () => {
    const index: VectorizeIndex = {
      upsert: async () => undefined,
      query: async () => ({
        matches: [{ id: 'unknown-doc', score: 0.7, metadata: { text: 'reconstructed' } }],
      }),
      deleteByIds: async () => undefined,
    };
    const store = new VectorizeVectorStore({ index, embeddingModel: new FakeEmbeddingModel() });
    const hits = await store.similaritySearch(searchRequest({ query: 'x', topK: 5 }));
    expect(hits[0]?.id).toBe('unknown-doc');
    expect(hits[0]?.text).toBe('reconstructed');
  });

  test('handles empty matches array from the index', async () => {
    const index: VectorizeIndex = {
      upsert: async () => undefined,
      query: async () => ({ matches: undefined }),
      deleteByIds: async () => undefined,
    };
    const store = new VectorizeVectorStore({ index, embeddingModel: new FakeEmbeddingModel() });
    const hits = await store.similaritySearch(searchRequest({ query: 'x', topK: 5 }));
    expect(hits).toEqual([]);
  });
});
