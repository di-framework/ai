import { Database } from 'bun:sqlite';
import { describe, expect, test } from 'bun:test';
import { mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  BunSqliteVectorStore,
  chatClientRequest,
  document,
  PrecomputedEmbeddingModel,
  Prompt,
  SimpleVectorStore,
  type VectorStore,
} from '@di-framework/ai';
import { InMemoryRepository } from '@di-framework/repo';
import {
  agentSkill,
  buildSkillsIndex,
  chunkRecordId,
  createSkillsToolboxAsync,
  loadSkillsIndex,
  SkillAdapterError,
  type SkillIndexWriteRequest,
  SkillSearchConnection,
  SkillSearchIndexer,
  SkillSearchRepository,
  type SkillSearchStorageAdapter,
  type SkillSearchStorageRecord,
  SkillsToolbox,
  toSkillIndexWriteRequest,
} from '../src/index.ts';

const metadata = {
  indexVersion: 'index-1',
  catalogVersion: 'catalog-1',
  dimensions: 2,
  model: 'model',
  revision: 'revision',
  embedderId: 'model@revision',
  scoring: 'cosine',
} as const;

const writeRequest: SkillIndexWriteRequest = {
  metadata,
  vectors: [
    { name: 'alpha', description: 'Alpha', chunk: 0, source: 'document', embedding: [1, 0] },
    { name: 'beta', description: 'Beta', chunk: 0, source: 'document', embedding: [0, 1] },
  ],
};

describe('SkillSearchRepository and SkillSearchIndexer', () => {
  test('round-trips through SimpleVectorStore, BunSqliteVectorStore, and a repo adapter', async () => {
    const simple = SkillSearchConnection.fromVectorStore(
      SimpleVectorStore.of(new PrecomputedEmbeddingModel(2)),
    );
    const sqlite = SkillSearchConnection.fromVectorStore(
      new BunSqliteVectorStore({
        db: new Database(':memory:'),
        embeddingModel: new PrecomputedEmbeddingModel(2),
        searchMode: 'ann',
      }),
    );
    const repo = SkillSearchConnection.fromStorageAdapter(
      new InMemoryRepository<SkillSearchStorageRecord, string>(),
    );
    for (const connection of [simple, sqlite, repo]) {
      const indexer = new SkillSearchIndexer(writeRequest, connection);
      const receipt = await indexer.replace();
      expect(receipt).toMatchObject({ ready: true, writtenVectors: 2 });
      const search = new SkillSearchRepository(connection);
      expect(await search.metadata()).toMatchObject({ indexVersion: 'index-1', ready: true });
      expect(await search.health()).toMatchObject({ status: 'ready', checkedVersion: 'index-1' });
      expect(await search.query([1, 0], { limit: 1 })).toEqual([
        { name: 'alpha', description: 'Alpha', score: 1, chunk: 0, source: 'document' },
      ]);
      expect((await search.query([0, 1], { minScore: 0.5 }))[0]?.name).toBe('beta');
      await expect(search.query([1, 0], { catalogVersion: 'stale' })).rejects.toMatchObject({
        code: 'STALE_CATALOG',
      });
      await expect(search.query([1, 0], { model: 'other' })).rejects.toMatchObject({
        code: 'MODEL_MISMATCH',
      });
    }
    expect(chunkRecordId('ns', 'alpha', 0)).toContain('alpha');
  });

  test('replace and upsert isolate namespaces and keep prompts to names/descriptions', async () => {
    const connection = SkillSearchConnection.fromVectorStore(
      SimpleVectorStore.of(new PrecomputedEmbeddingModel(2)),
    );
    const indexer = new SkillSearchIndexer(connection);
    await indexer.replace({
      metadata: { ...metadata, namespace: 'a' },
      vectors: [
        {
          name: 'alpha',
          description: 'Handle alpha tasks',
          chunk: 0,
          source: 'document',
          embedding: [1, 0],
        },
      ],
    });
    await indexer.upsert({
      metadata: { ...metadata, namespace: 'b', indexVersion: 'index-b' },
      vectors: [
        {
          name: 'beta',
          description: 'Handle beta tasks',
          chunk: 0,
          source: 'document',
          embedding: [0, 1],
        },
      ],
    });
    const search = new SkillSearchRepository(connection);
    expect((await search.query([1, 0], { namespace: 'a', limit: 1 }))[0]?.name).toBe('alpha');
    expect((await search.query([0, 1], { namespace: 'b', limit: 1 }))[0]?.name).toBe('beta');
    await indexer.replace({
      metadata: { ...metadata, namespace: 'a', indexVersion: 'index-2' },
      vectors: [
        {
          name: 'gamma',
          description: 'Handle gamma tasks',
          chunk: 0,
          source: 'document',
          embedding: [1, 0],
        },
      ],
    });
    expect((await search.query([1, 0], { namespace: 'a', limit: 1 }))[0]?.name).toBe('gamma');

    const skill = agentSkill({
      name: 'gamma',
      description: 'Handle gamma tasks',
      content: 'SECRET BODY',
    });
    const toolbox = await createSkillsToolboxAsync({
      workspace: process.cwd(),
      semanticDiscovery: {
        catalogStore: {
          capabilities: {
            namespaces: true,
            lazyBodies: true,
            vectorSearch: false,
            indexWriting: false,
            eventuallyConsistent: false,
          },
          list: async () => [
            { name: 'gamma', description: skill.description, sourceHash: 'hash', version: 'v1' },
          ],
          load: async () => skill,
          version: async () => 'catalog-1',
          health: async () => ({ status: 'ready' as const, checkedVersion: 'catalog-1' }),
        },
        vectorSearch: search,
        embedder: {
          id: 'model@revision',
          model: 'model',
          revision: 'revision',
          embed: async (texts) => texts.map(() => new Float32Array([1, 0])),
          split: async (text) => [text],
        },
        namespace: 'a',
        limit: 1,
      },
      todos: false,
      list: false,
      glob: false,
      grep: false,
    });
    const selected = await toolbox.retrievalAdvisor?.before(
      chatClientRequest(new Prompt('please do a gamma task', { toolCallbacks: toolbox.tools })),
    );
    const description = selected?.prompt.options?.toolCallbacks?.find(
      (tool) => tool.toolDefinition.name === 'Skill',
    )?.toolDefinition.description;
    expect(description).toContain('<name>gamma</name>');
    expect(description).not.toContain('SECRET BODY');
    expect(JSON.stringify(selected?.prompt)).not.toContain('index-2');
    expect(SkillsToolbox).toBeDefined();
  });

  test('vector-backed indexing stores full source chunks without exposing them to selection', async () => {
    const store = new TrackingStore();
    const connection = SkillSearchConnection.fromVectorStore(store);
    const secretBody = 'INTERNAL FULL SKILL INSTRUCTIONS';
    const skill = agentSkill({
      name: 'full-source',
      description: 'Choose this skill for source-aware work',
      content: secretBody,
    });

    await buildSkillsIndex({
      skills: [skill],
      threshold: 0,
      embedder: {
        id: 'local@test',
        model: 'local',
        revision: 'test',
        embed: async (texts) => texts.map(() => new Float32Array([1, 0])),
        split: async (text) => [text],
      },
      writer: new SkillSearchIndexer(connection),
    });

    const chunk = store.documents().find((doc) => doc.metadata.kind === 'skill-chunk');
    expect(chunk?.text).toContain(secretBody);

    const hits = await new SkillSearchRepository(connection).query([1, 0], { limit: 1 });
    expect(hits).toEqual([
      {
        name: 'full-source',
        description: 'Choose this skill for source-aware work',
        score: 1,
        chunk: 0,
        source: 'document',
      },
    ]);
    expect(JSON.stringify(hits)).not.toContain(secretBody);
  });

  test('fails closed for not-ready indexes, bad queries, and corrupt metadata', async () => {
    const store = SimpleVectorStore.of(new PrecomputedEmbeddingModel(2));
    const connection = SkillSearchConnection.fromVectorStore(store);
    const search = new SkillSearchRepository(connection);
    await expect(search.metadata()).rejects.toMatchObject({ code: 'NOT_READY' });
    expect(await search.health()).toMatchObject({ status: 'not-ready' });
    await expect(
      new SkillSearchRepository(
        SkillSearchConnection.fromVectorStore({
          add: async () => undefined,
          delete: async () => undefined,
          get: async () => {
            throw new Error('meta failed');
          },
          similaritySearch: async () => [],
        }),
      ).health(),
    ).rejects.toMatchObject({ code: 'INVALID_RESPONSE' });
    await expect(new SkillSearchIndexer(connection).replace()).rejects.toMatchObject({
      code: 'INVALID_RESPONSE',
    });

    await new SkillSearchIndexer(connection).replace(writeRequest);
    await expect(search.query([1, 0, 0])).rejects.toMatchObject({ code: 'INVALID_RESPONSE' });
    await expect(search.query([1, 0], { limit: 0 })).rejects.toMatchObject({
      code: 'INVALID_RESPONSE',
    });
    await expect(search.query([1, 0], { minScore: Number.NaN })).rejects.toMatchObject({
      code: 'INVALID_RESPONSE',
    });

    await store.add([
      document({
        id: 'meta:',
        text: 'broken',
        embedding: [0, 0],
        metadata: { kind: 'skill-index-meta', namespace: '' },
      }),
    ]);
    await expect(search.health()).rejects.toMatchObject({ code: 'INVALID_RESPONSE' });
    expect(new SkillAdapterError('INVALID_RESPONSE', 'x')).toMatchObject({
      name: 'SkillAdapterError',
    });
  });

  test('vector stores without get or deleteByFilter still write through tracked ids', async () => {
    const missingGet = SkillSearchConnection.fromVectorStore({
      add: async () => undefined,
      delete: async () => undefined,
      similaritySearch: async () => [],
    } satisfies VectorStore);
    await expect(new SkillSearchRepository(missingGet).health()).rejects.toMatchObject({
      code: 'INVALID_RESPONSE',
    });

    const tracked = new TrackingStore();
    const connection = SkillSearchConnection.fromVectorStore(tracked);
    await new SkillSearchIndexer(connection).replace(writeRequest);
    expect(tracked.ids().some((id) => id.includes('alpha'))).toBe(true);
    await new SkillSearchIndexer(connection).replace({
      metadata,
      vectors: [
        { name: 'gamma', description: 'Gamma', chunk: 0, source: 'document', embedding: [1, 0] },
      ],
    });
    expect(tracked.ids().some((id) => id.includes('alpha'))).toBe(false);
    expect((await new SkillSearchRepository(connection).query([1, 0], { limit: 1 }))[0]?.name).toBe(
      'gamma',
    );
  });

  test('storage adapters use transactions and exact cosine, including zero vectors', async () => {
    const adapter = new MemoryAdapter();
    const connection = SkillSearchConnection.fromStorageAdapter(adapter);
    await connection.saveMetadata({ ...metadata, ready: true, dimensions: 0 });
    await new SkillSearchIndexer(connection).replace({
      metadata,
      vectors: [
        { name: 'zero', description: 'Zero', chunk: 0, source: 'document', embedding: [0, 0] },
        { name: 'east', description: 'East', chunk: 0, source: 'document', embedding: [1, 0] },
      ],
    });
    expect(adapter.transactions).toBeGreaterThan(0);
    const hits = await new SkillSearchRepository(connection).query([1, 0], { limit: 2 });
    expect(hits[0]?.name).toBe('east');
    expect(hits[1]?.score).toBe(0);
    await connection.upsertChunks([
      {
        id: chunkRecordId('', 'west', 0),
        kind: 'skill-chunk',
        name: 'west',
        description: 'West',
        chunk: 0,
        source: 'document',
        namespace: '',
        embedding: [-1, 0],
      },
    ]);
    expect((await connection.queryByEmbedding({ vector: [-1, 0], limit: 1 }))[0]?.name).toBe(
      'west',
    );
  });

  test('wraps backend failures and converts JSONL indexes into write requests', async () => {
    const writeBoom: VectorStore = {
      add: async () => {
        throw new Error('write failed');
      },
      delete: async () => undefined,
      get: async () => null,
      similaritySearch: async () => [],
    };
    await expect(
      new SkillSearchIndexer(SkillSearchConnection.fromVectorStore(writeBoom)).replace(
        writeRequest,
      ),
    ).rejects.toMatchObject({ code: 'INVALID_RESPONSE' });

    const searchBoom: VectorStore = {
      add: async () => undefined,
      delete: async () => undefined,
      get: async () =>
        document({
          id: 'meta:',
          text: 'index-1',
          embedding: [0, 0],
          metadata: { ...metadata, ready: true, kind: 'skill-index-meta', namespace: '' },
        }),
      similaritySearch: async () => {
        throw new Error('search failed');
      },
    };
    await expect(
      new SkillSearchRepository(SkillSearchConnection.fromVectorStore(searchBoom)).query([1, 0]),
    ).rejects.toMatchObject({ code: 'INVALID_RESPONSE' });

    const directory = join(tmpdir(), `skill-search-${Date.now()}`);
    mkdirSync(directory, { recursive: true });
    const built = await buildSkillsIndex({
      skills: [
        agentSkill({
          name: 'review',
          description: 'Review typescript',
          content: '# review',
        }),
      ],
      threshold: 0,
      embedder: {
        id: 'local@test',
        model: 'local',
        revision: 'test',
        embed: async (texts) => texts.map(() => new Float32Array([1, 0])),
        split: async (text) => [text],
      },
      outputFile: join(directory, 'skills.jsonl'),
    });
    const request = toSkillIndexWriteRequest(loadSkillsIndex(built.outputFile));
    expect(request.vectors.length).toBeGreaterThan(0);
    expect(request.metadata.dimensions).toBe(2);
  });
});

class TrackingStore implements VectorStore {
  private readonly docs = new Map<string, ReturnType<typeof document>>();

  async add(documents: readonly ReturnType<typeof document>[]): Promise<void> {
    for (const doc of documents) this.docs.set(doc.id, doc);
  }

  async delete(ids: readonly string[]): Promise<void> {
    for (const id of ids) this.docs.delete(id);
  }

  async get(id: string) {
    return this.docs.get(id) ?? null;
  }

  async similaritySearch(request: {
    queryEmbedding?: ArrayLike<number>;
    topK: number;
    similarityThreshold: number;
  }) {
    const query = request.queryEmbedding ?? [0, 0];
    return [...this.docs.values()]
      .filter((doc) => doc.metadata.kind === 'skill-chunk')
      .map((doc) => {
        const embedding = (doc.embedding ?? [0, 0]) as number[];
        let dot = 0;
        let left = 0;
        let right = 0;
        for (let index = 0; index < query.length; index++) {
          const a = Number(query[index]);
          const b = Number(embedding[index] ?? 0);
          dot += a * b;
          left += a * a;
          right += b * b;
        }
        const score = left && right ? dot / Math.sqrt(left * right) : 0;
        return { ...doc, score };
      })
      .filter((doc) => (doc.score ?? 0) >= request.similarityThreshold)
      .sort((left, right) => (right.score ?? 0) - (left.score ?? 0))
      .slice(0, request.topK);
  }

  ids(): string[] {
    return [...this.docs.keys()];
  }

  documents(): ReturnType<typeof document>[] {
    return [...this.docs.values()];
  }
}

class MemoryAdapter implements SkillSearchStorageAdapter {
  readonly items = new Map<string, SkillSearchStorageRecord>();
  transactions = 0;

  async findAll(): Promise<SkillSearchStorageRecord[]> {
    return [...this.items.values()];
  }

  async save(entity: SkillSearchStorageRecord): Promise<SkillSearchStorageRecord> {
    this.items.set(entity.id, entity);
    return entity;
  }

  async delete(id: string): Promise<boolean> {
    return this.items.delete(id);
  }

  async transaction<T>(fn: (adapter: this) => Promise<T>): Promise<T> {
    this.transactions += 1;
    return fn(this);
  }
}
