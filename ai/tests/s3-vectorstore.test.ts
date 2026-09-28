import { describe, expect, test } from 'bun:test';
import { Container } from '@di-framework/core/container';
import { VectorStore } from '../src/di/annotations/rag.ts';
import { configureAi } from '../src/di/register.ts';
import { AiTokens } from '../src/di/tokens.ts';
import { textDocument } from '../src/document/document.ts';
import { FakeEmbeddingModel } from '../src/embedding/fake-embedding-model.ts';
import {
  cosineSimilarity,
  InMemoryS3VectorsClient,
  matchesS3Filter,
  S3VectorStore,
  type S3VectorsClient,
  translateS3FilterExpression,
} from '../src/vectorstore/adapters/s3.ts';
import {
  FilterExpressionBuilder,
  filterExpression,
  filterGroup,
  filterKey,
  filterValue,
} from '../src/vectorstore/filter/index.ts';
import { searchRequest } from '../src/vectorstore/search-request.ts';

describe('S3VectorStore and Filter Translation', () => {
  describe('translateS3FilterExpression', () => {
    test('returns null for null or undefined expression', () => {
      expect(translateS3FilterExpression(null)).toBeNull();
      expect(translateS3FilterExpression(undefined)).toBeNull();
    });

    test('translates comparison operators and handles quoted keys', () => {
      const b = new FilterExpressionBuilder();

      expect(translateS3FilterExpression(b.eq('author', 'Alice').build())).toEqual({
        author: { $eq: 'Alice' },
      });
      expect(translateS3FilterExpression(b.eq('"author"', 'Bob').build())).toEqual({
        author: { $eq: 'Bob' },
      });
      expect(translateS3FilterExpression(b.eq("'author'", 'Charlie').build())).toEqual({
        author: { $eq: 'Charlie' },
      });
      expect(translateS3FilterExpression(b.ne('status', 'draft').build())).toEqual({
        status: { $ne: 'draft' },
      });
      expect(translateS3FilterExpression(b.gt('score', 80).build())).toEqual({
        score: { $gt: 80 },
      });
      expect(translateS3FilterExpression(b.gte('score', 90).build())).toEqual({
        score: { $gte: 90 },
      });
      expect(translateS3FilterExpression(b.lt('age', 30).build())).toEqual({
        age: { $lt: 30 },
      });
      expect(translateS3FilterExpression(b.lte('age', 25).build())).toEqual({
        age: { $lte: 25 },
      });
      expect(translateS3FilterExpression(b.in('category', 'tech', 'science').build())).toEqual({
        category: { $in: ['tech', 'science'] },
      });
      expect(translateS3FilterExpression(b.nin('tag', 'deprecated', 'legacy').build())).toEqual({
        tag: { $nin: ['deprecated', 'legacy'] },
      });
      expect(translateS3FilterExpression(b.isNull('deletedAt').build())).toEqual({
        deletedAt: { $exists: false },
      });
      expect(translateS3FilterExpression(b.isNotNull('publishedAt').build())).toEqual({
        publishedAt: { $exists: true },
      });
    });

    test('translates logical operators AND, OR, NOT and groups', () => {
      const b = new FilterExpressionBuilder();
      const andExpr = b.and(b.eq('dept', 'eng'), b.gte('level', 3)).build();
      expect(translateS3FilterExpression(andExpr)).toEqual({
        $and: [{ dept: { $eq: 'eng' } }, { level: { $gte: 3 } }],
      });

      const orExpr = b.or(b.eq('role', 'admin'), b.eq('role', 'root')).build();
      expect(translateS3FilterExpression(orExpr)).toEqual({
        $or: [{ role: { $eq: 'admin' } }, { role: { $eq: 'root' } }],
      });

      const notExpr = b.not(b.eq('archived', true)).build();
      expect(translateS3FilterExpression(notExpr)).toEqual({
        $not: { archived: { $eq: true } },
      });

      const groupExpr = b.group(b.eq('active', true)).build();
      expect(translateS3FilterExpression(groupExpr)).toEqual({
        active: { $eq: true },
      });
    });

    test('handles naked FilterOperand expressions and IN with single value', () => {
      const singleIn = filterExpression('IN', filterKey('type'), filterValue('book'));
      expect(translateS3FilterExpression(singleIn)).toEqual({
        type: { $in: ['book'] },
      });

      const singleNin = filterExpression('NIN', filterKey('type'), filterValue('magazine'));
      expect(translateS3FilterExpression(singleNin)).toEqual({
        type: { $nin: ['magazine'] },
      });

      const nestedOperand = filterExpression(
        'AND',
        filterGroup(filterExpression('EQ', filterKey('k'), filterValue('v'))),
        filterExpression('EQ', filterKey('k2'), filterValue('v2')),
      );
      expect(translateS3FilterExpression(nestedOperand)).toEqual({
        $and: [{ k: { $eq: 'v' } }, { k2: { $eq: 'v2' } }],
      });
    });

    test('throws error on unsupported filter expression type', () => {
      const invalid = filterExpression('UNKNOWN' as any, filterKey('k'), filterValue('v'));
      expect(() => translateS3FilterExpression(invalid)).toThrow(
        'Unsupported S3 filter expression type: UNKNOWN',
      );
    });
  });

  describe('cosineSimilarity', () => {
    test('calculates correct cosine similarity', () => {
      expect(cosineSimilarity([1, 0, 0], [1, 0, 0])).toBeCloseTo(1.0);
      expect(cosineSimilarity([1, 0, 0], [0, 1, 0])).toBeCloseTo(0.0);
      expect(cosineSimilarity([1, 1], [-1, -1])).toBeCloseTo(-1.0);
      expect(cosineSimilarity([0, 0], [0, 0])).toBe(0);
      expect(cosineSimilarity([], [])).toBe(0);
      expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBeCloseTo(1.0);
      expect(() => cosineSimilarity([1, 2, 3], [1, 2])).toThrow('embedding dimension mismatch');
    });
  });

  describe('matchesS3Filter', () => {
    test('evaluates various filter conditions against metadata', () => {
      expect(matchesS3Filter({}, null)).toBe(true);
      expect(matchesS3Filter({}, undefined)).toBe(true);
      expect(matchesS3Filter({ role: 'admin' }, { role: 'admin' })).toBe(true);
      expect(matchesS3Filter({ role: 'user' }, { role: 'admin' })).toBe(false);

      const meta = {
        name: 'Alice',
        age: 30,
        tags: ['di', 'typescript'],
        verified: true,
        score: 95.5,
      };

      expect(matchesS3Filter(meta, { age: { $eq: 30 } })).toBe(true);
      expect(matchesS3Filter(meta, { age: { $ne: 25 } })).toBe(true);
      expect(matchesS3Filter(meta, { age: { $gt: 20 } })).toBe(true);
      expect(matchesS3Filter(meta, { age: { $gt: 35 } })).toBe(false);
      expect(matchesS3Filter(meta, { age: { $gte: 30 } })).toBe(true);
      expect(matchesS3Filter(meta, { age: { $lt: 40 } })).toBe(true);
      expect(matchesS3Filter(meta, { age: { $lt: 25 } })).toBe(false);
      expect(matchesS3Filter(meta, { age: { $lte: 30 } })).toBe(true);
      expect(matchesS3Filter(meta, { name: { $in: ['Alice', 'Bob'] } })).toBe(true);
      expect(matchesS3Filter(meta, { name: { $in: ['Charlie', 'Dave'] } })).toBe(false);
      expect(matchesS3Filter(meta, { name: { $nin: ['Eve', 'Mallory'] } })).toBe(true);
      expect(matchesS3Filter(meta, { name: { $nin: ['Alice', 'Bob'] } })).toBe(false);
      expect(matchesS3Filter(meta, { verified: { $exists: true } })).toBe(true);
      expect(matchesS3Filter(meta, { missing: { $exists: false } })).toBe(true);
      expect(matchesS3Filter(meta, { verified: { $exists: false } })).toBe(false);

      // Logical combinations
      expect(
        matchesS3Filter(meta, {
          $and: [{ age: { $gte: 25 } }, { name: { $eq: 'Alice' } }],
        }),
      ).toBe(true);
      expect(
        matchesS3Filter(meta, {
          $or: [{ age: { $lt: 20 } }, { name: { $eq: 'Alice' } }],
        }),
      ).toBe(true);
      expect(
        matchesS3Filter(meta, {
          $not: { age: { $lt: 20 } },
        }),
      ).toBe(true);
    });
  });

  describe('InMemoryS3VectorsClient', () => {
    test('handles full CRUD lifecycle in memory', async () => {
      const client = new InMemoryS3VectorsClient();
      await client.putVectors({
        vectorBucketName: 'test-bucket',
        indexName: 'test-index',
        vectors: [
          { id: 'v1', vector: [1, 0], document: 'doc 1', metadata: { cat: 'a' } },
          { id: 'v2', vector: [0, 1], document: 'doc 2', metadata: { cat: 'b' } },
        ],
      });

      const getResult = await client.getVector({
        vectorBucketName: 'test-bucket',
        indexName: 'test-index',
        id: 'v1',
      });
      expect(getResult?.id).toBe('v1');
      expect(getResult?.document).toBe('doc 1');

      const queryResult = await client.queryVectors({
        vectorBucketName: 'test-bucket',
        indexName: 'test-index',
        queryVector: [1, 0],
        topK: 1,
        returnDocument: true,
        returnMetadata: true,
      });
      expect(queryResult.vectors?.[0]?.id).toBe('v1');
      expect(queryResult.vectors?.[0]?.score).toBeCloseTo(1.0);

      // Delete by filter
      await client.deleteVectors({
        vectorBucketName: 'test-bucket',
        indexName: 'test-index',
        filter: { cat: { $eq: 'a' } },
      });
      expect(
        await client.getVector({
          vectorBucketName: 'test-bucket',
          indexName: 'test-index',
          id: 'v1',
        }),
      ).toBeNull();
    });
  });

  describe('S3VectorStore', () => {
    test('implements VectorStore with default InMemoryS3VectorsClient', async () => {
      const embeddingModel = new FakeEmbeddingModel({ dimensions: 4 });
      const store = new S3VectorStore({
        vectorBucketName: 'my-bucket',
        indexName: 'my-index',
        region: 'us-east-1',
        embeddingModel,
      });

      expect(store.name).toBe('S3VectorStore');
      expect(store.vectorBucketName).toBe('my-bucket');
      expect(store.indexName).toBe('my-index');
      expect(store.region).toBe('us-east-1');

      const doc1 = textDocument(
        'AWS S3 Vectors for serverless vector retrieval',
        { topic: 's3' },
        'doc-1',
      );
      const doc2 = textDocument(
        'PostgreSQL pgvector for relational vectors',
        { topic: 'pg' },
        'doc-2',
      );

      await store.add([doc1, doc2]);

      // Point lookup
      const fetched1 = await store.get('doc-1');
      expect(fetched1?.id).toBe('doc-1');
      expect(fetched1?.text).toBe('AWS S3 Vectors for serverless vector retrieval');
      expect(fetched1?.metadata).toEqual({
        topic: 's3',
        text: 'AWS S3 Vectors for serverless vector retrieval',
      });

      // Similarity search
      const hits = await store.similaritySearch(
        searchRequest({
          query: 'AWS S3 Vectors',
          topK: 2,
        }),
      );
      expect(hits.length).toBeGreaterThan(0);
      expect(hits[0]?.id).toBe('doc-1');

      // Similarity search with query string
      const hitsQuery = await store.similaritySearchQuery('AWS S3 Vectors');
      expect(hitsQuery.length).toBeGreaterThan(0);

      // Filtered similarity search
      const b = new FilterExpressionBuilder();
      const filteredHits = await store.similaritySearch(
        searchRequest({
          query: 'vectors',
          topK: 5,
          filterExpression: b.eq('topic', 'pg').build(),
        }),
      );
      expect(filteredHits.length).toBe(1);
      expect(filteredHits[0]?.id).toBe('doc-2');

      // Threshold pruning
      const prunedHits = await store.similaritySearch(
        searchRequest({
          query: 'AWS S3 Vectors',
          similarityThreshold: 0.9999,
        }),
      );
      // Only close matches
      expect(prunedHits.length).toBeLessThanOrEqual(2);

      // Delete by ID
      await store.delete(['doc-2']);
      expect(await store.get('doc-2')).toBeNull();

      // Delete by filter
      await store.deleteByFilter(b.eq('topic', 's3').build());
      expect(await store.get('doc-1')).toBeNull();
    });

    test('supports custom S3VectorsClient and edge cases', async () => {
      const putCalls: any[] = [];
      const queryCalls: any[] = [];
      const deleteCalls: any[] = [];

      const customClient: S3VectorsClient = {
        async putVectors(input) {
          putCalls.push(input);
        },
        async queryVectors(input) {
          queryCalls.push(input);
          return {
            vectors: [
              {
                id: 'remote-1',
                score: 0.95,
                document: 'Remote text',
                metadata: { source: 'remote' },
              },
              {
                id: 'remote-low',
                score: 0.1,
                document: 'Low score text',
                metadata: { source: 'low' },
              },
            ],
          };
        },
        async deleteVectors(input) {
          deleteCalls.push(input);
        },
        async getVector(input) {
          if (input.id === 'remote-1') {
            return {
              id: 'remote-1',
              document: 'Remote text',
              metadata: { source: 'remote' },
            };
          }
          return null;
        },
      };

      const store = new S3VectorStore({
        vectorBucketName: 'custom-bucket',
        indexName: 'custom-index',
        name: 'CustomS3Store',
        client: customClient,
        embeddingModel: new FakeEmbeddingModel(),
      });

      expect(store.name).toBe('CustomS3Store');

      await store.add([textDocument('hello custom', { cat: 'x' }, 'c1')]);
      expect(putCalls.length).toBe(1);
      expect(putCalls[0].vectorBucketName).toBe('custom-bucket');

      const fetched = await store.get('remote-1');
      expect(fetched?.id).toBe('remote-1');
      expect(fetched?.text).toBe('Remote text');

      const notFound = await store.get('missing-remote');
      expect(notFound).toBeNull();

      const results = await store.similaritySearch(
        searchRequest({
          query: 'custom search',
          topK: 5,
          similarityThreshold: 0.5,
        }),
      );
      expect(results.length).toBe(1);
      expect(results[0]?.id).toBe('remote-1');
      expect(results[0]?.score).toBe(0.95);

      await store.delete(['c1']);
      expect(deleteCalls.length).toBe(1);
    });

    test('handles client without getVector method', async () => {
      const clientWithoutGet: S3VectorsClient = {
        async putVectors() {},
        async queryVectors() {
          return { vectors: [] };
        },
        async deleteVectors() {},
      };

      const store = new S3VectorStore({
        vectorBucketName: 'b',
        indexName: 'i',
        client: clientWithoutGet,
        embeddingModel: new FakeEmbeddingModel(),
      });

      await store.add([textDocument('local fallback', {}, 'loc1')]);
      expect(await store.get('loc1')).toMatchObject({ id: 'loc1', text: 'local fallback' });
      expect(await store.get('missing')).toBeNull();
    });

    test('supports DI container registration and configureAi', async () => {
      const container = new Container();
      const embeddingModel = new FakeEmbeddingModel();
      const s3Store = new S3VectorStore({
        vectorBucketName: 'di-bucket',
        indexName: 'di-index',
        embeddingModel,
      });

      configureAi({
        container,
        chatModel: {
          call: async () => ({
            text: 'response',
            message: { role: 'assistant', text: 'response' },
            finishReason: 'stop',
            usage: {},
          }),
          stream: async function* () {},
        } as any,
        vectorStore: s3Store,
      });

      const resolved = container.resolve<S3VectorStore>(AiTokens.VECTOR_STORE);
      expect(resolved).toBe(s3Store);
      expect(resolved.name).toBe('S3VectorStore');
    });
  });
});
