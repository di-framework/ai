import {
  document,
  FilterExpressionBuilder,
  searchRequest,
  type VectorStore,
} from '@di-framework/ai';
import {
  SkillAdapterError,
  type SkillChunkSource,
  type SkillVectorIndexMetadata,
  validateWrite,
} from './skill-adapters.ts';

const KIND_CHUNK = 'skill-chunk';
const KIND_META = 'skill-index-meta';
const filters = new FilterExpressionBuilder();

export interface SkillSearchChunkRecord {
  readonly id: string;
  readonly kind: typeof KIND_CHUNK;
  readonly name: string;
  readonly description: string;
  readonly text?: string;
  readonly chunk: number;
  readonly source: SkillChunkSource;
  readonly namespace: string;
  readonly documentHash?: string;
  readonly embedding: ArrayLike<number>;
}

export interface SkillSearchHit {
  readonly name: string;
  readonly description: string;
  readonly chunk: number;
  readonly source: SkillChunkSource;
  readonly score: number;
}

export interface SkillSearchQuery {
  readonly vector: ArrayLike<number>;
  readonly limit: number;
  readonly minScore?: number;
  readonly namespace?: string;
}

/**
 * Structural subset of `@di-framework/repo` `StorageAdapter`.
 * `InMemoryRepository` and SQL adapters work when the entity is {@link SkillSearchStorageRecord}.
 */
export interface SkillSearchStorageAdapter {
  findAll(): Promise<readonly SkillSearchStorageRecord[] | SkillSearchStorageRecord[]>;
  save(entity: SkillSearchStorageRecord): Promise<SkillSearchStorageRecord>;
  delete(id: string): Promise<boolean>;
  transaction?<T>(fn: (adapter: this) => Promise<T>): Promise<T>;
}

export interface SkillSearchStorageRecord {
  id: string;
  kind: typeof KIND_CHUNK | typeof KIND_META;
  name: string;
  description: string;
  chunk: number;
  source: SkillChunkSource;
  namespace: string;
  documentHash?: string;
  embedding: number[];
  metadata?: SkillVectorIndexMetadata;
}

interface SkillSearchBackend {
  loadMetadata(namespace?: string): Promise<SkillVectorIndexMetadata | undefined>;
  saveMetadata(metadata: SkillVectorIndexMetadata): Promise<void>;
  replaceChunks(
    namespace: string | undefined,
    records: readonly SkillSearchChunkRecord[],
  ): Promise<void>;
  upsertChunks(records: readonly SkillSearchChunkRecord[]): Promise<void>;
  queryByEmbedding(query: SkillSearchQuery): Promise<readonly SkillSearchHit[]>;
}

export class SkillSearchConnection implements SkillSearchBackend {
  private constructor(private readonly impl: SkillSearchBackend) {}

  static fromVectorStore(store: VectorStore): SkillSearchConnection {
    return new SkillSearchConnection(new VectorStoreConnection(store));
  }

  static fromStorageAdapter(adapter: SkillSearchStorageAdapter): SkillSearchConnection {
    return new SkillSearchConnection(new StorageAdapterConnection(adapter));
  }

  loadMetadata(namespace?: string): Promise<SkillVectorIndexMetadata | undefined> {
    return this.impl.loadMetadata(namespace);
  }

  saveMetadata(metadata: SkillVectorIndexMetadata): Promise<void> {
    return this.impl.saveMetadata(metadata);
  }

  replaceChunks(
    namespace: string | undefined,
    records: readonly SkillSearchChunkRecord[],
  ): Promise<void> {
    return this.impl.replaceChunks(namespace, records);
  }

  upsertChunks(records: readonly SkillSearchChunkRecord[]): Promise<void> {
    return this.impl.upsertChunks(records);
  }

  queryByEmbedding(query: SkillSearchQuery): Promise<readonly SkillSearchHit[]> {
    return this.impl.queryByEmbedding(query);
  }
}

export function chunkRecordId(namespace: string | undefined, name: string, chunk: number): string {
  return `${namespaceKey(namespace)}\0${name}\0${chunk}`;
}

export function toChunkRecords(
  request: Parameters<typeof validateWrite>[0],
): SkillSearchChunkRecord[] {
  validateWrite(request);
  const namespace = namespaceKey(request.metadata.namespace);
  return request.vectors.map((vector) => ({
    id: chunkRecordId(namespace, vector.name, vector.chunk),
    kind: KIND_CHUNK,
    name: vector.name,
    description: vector.description,
    text: vector.text,
    chunk: vector.chunk,
    source: vector.source,
    namespace,
    documentHash: vector.documentHash,
    embedding: vector.embedding,
  }));
}

function namespaceKey(namespace?: string): string {
  return namespace ?? '';
}

function metaId(namespace?: string): string {
  return `meta:${namespaceKey(namespace)}`;
}

class VectorStoreConnection implements SkillSearchBackend {
  constructor(private readonly store: VectorStore) {}

  async loadMetadata(namespace?: string): Promise<SkillVectorIndexMetadata | undefined> {
    const raw = await this.metaDocument(namespace);
    return raw ? readMetadata(raw.metadata) : undefined;
  }

  async saveMetadata(metadata: SkillVectorIndexMetadata): Promise<void> {
    const ns = namespaceKey(metadata.namespace);
    const previous = await this.metaDocument(ns);
    await this.writeMeta(metadata, chunkIdList(previous?.metadata));
  }

  async replaceChunks(
    namespace: string | undefined,
    records: readonly SkillSearchChunkRecord[],
  ): Promise<void> {
    const ns = namespaceKey(namespace);
    if (this.store.deleteByFilter) {
      await this.store.deleteByFilter(
        filters.and(filters.eq('kind', KIND_CHUNK), filters.eq('namespace', ns)).build(),
      );
    } else {
      const previousIds = chunkIdList((await this.metaDocument(ns))?.metadata);
      if (previousIds.length > 0) await this.store.delete(previousIds);
    }
    await this.writeChunks(records);
    const metadata = await this.loadMetadata(ns);
    if (metadata)
      await this.writeMeta(
        metadata,
        records.map((record) => record.id),
      );
  }

  async upsertChunks(records: readonly SkillSearchChunkRecord[]): Promise<void> {
    await this.writeChunks(records);
    for (const [namespace, group] of groupByNamespace(records)) {
      const existing = new Set(chunkIdList((await this.metaDocument(namespace))?.metadata));
      for (const record of group) existing.add(record.id);
      const metadata = await this.loadMetadata(namespace);
      if (metadata) await this.writeMeta(metadata, [...existing]);
    }
  }

  async queryByEmbedding(query: SkillSearchQuery): Promise<readonly SkillSearchHit[]> {
    const ns = namespaceKey(query.namespace);
    const minScore = query.minScore ?? Number.NEGATIVE_INFINITY;
    const docs = await this.store.similaritySearch(
      searchRequest({
        query: '',
        queryEmbedding: query.vector,
        topK: query.limit,
        similarityThreshold: 0,
        filterExpression: filters
          .and(filters.eq('kind', KIND_CHUNK), filters.eq('namespace', ns))
          .build(),
      }),
    );
    return docs
      .map((doc) => ({
        name: String(doc.metadata.name ?? ''),
        description: String(doc.metadata.description ?? ''),
        chunk: Number(doc.metadata.chunk ?? 0),
        source: (doc.metadata.source as SkillChunkSource) ?? 'document',
        score: doc.score ?? 0,
      }))
      .filter((hit) => hit.score >= minScore);
  }

  private async writeChunks(records: readonly SkillSearchChunkRecord[]): Promise<void> {
    if (records.length === 0) return;
    await this.store.add(
      records.map((record) =>
        document({
          id: record.id,
          text: record.text ?? record.description,
          embedding: record.embedding,
          metadata: {
            kind: KIND_CHUNK,
            name: record.name,
            description: record.description,
            chunk: record.chunk,
            source: record.source,
            namespace: record.namespace,
            documentHash: record.documentHash,
          },
        }),
      ),
    );
  }

  private async writeMeta(
    metadata: SkillVectorIndexMetadata,
    chunkIds: readonly string[],
  ): Promise<void> {
    await this.store.add([
      document({
        id: metaId(metadata.namespace),
        text: metadata.indexVersion,
        embedding: zeroEmbedding(metadata.dimensions),
        metadata: {
          kind: KIND_META,
          namespace: namespaceKey(metadata.namespace),
          ...metadata,
          chunkIds: [...chunkIds],
        },
      }),
    ]);
  }

  private async metaDocument(namespace?: string) {
    if (!this.store.get) {
      throw new SkillAdapterError(
        'INVALID_RESPONSE',
        'Vector store does not support point lookup for skill index metadata',
      );
    }
    return this.store.get(metaId(namespace));
  }
}

function chunkIdList(metadata?: Readonly<Record<string, unknown>>): string[] {
  const value = metadata?.chunkIds;
  return Array.isArray(value) ? value.map((id) => String(id)) : [];
}

class StorageAdapterConnection implements SkillSearchBackend {
  constructor(private readonly adapter: SkillSearchStorageAdapter) {}

  async loadMetadata(namespace?: string): Promise<SkillVectorIndexMetadata | undefined> {
    const ns = namespaceKey(namespace);
    const rows = await this.adapter.findAll();
    const meta = rows.find((row) => row.kind === KIND_META && row.namespace === ns);
    return meta?.metadata;
  }

  async saveMetadata(metadata: SkillVectorIndexMetadata): Promise<void> {
    const ns = namespaceKey(metadata.namespace);
    await this.adapter.save({
      id: metaId(ns),
      kind: KIND_META,
      name: '',
      description: '',
      chunk: 0,
      source: 'document',
      namespace: ns,
      embedding: zeroEmbedding(metadata.dimensions),
      metadata,
    });
  }

  async replaceChunks(
    namespace: string | undefined,
    records: readonly SkillSearchChunkRecord[],
  ): Promise<void> {
    const ns = namespaceKey(namespace);
    await this.mutate(async () => {
      const existing = await this.adapter.findAll();
      for (const row of existing) {
        if (row.namespace === ns && row.kind === KIND_CHUNK) {
          await this.adapter.delete(row.id);
        }
      }
      for (const record of records) {
        await this.adapter.save(storageRecord(record));
      }
    });
  }

  async upsertChunks(records: readonly SkillSearchChunkRecord[]): Promise<void> {
    await this.mutate(async () => {
      for (const record of records) await this.adapter.save(storageRecord(record));
    });
  }

  async queryByEmbedding(query: SkillSearchQuery): Promise<readonly SkillSearchHit[]> {
    const ns = namespaceKey(query.namespace);
    const minScore = query.minScore ?? Number.NEGATIVE_INFINITY;
    const rows = await this.adapter.findAll();
    return rows
      .filter((row) => row.kind === KIND_CHUNK && row.namespace === ns)
      .map((row) => ({
        name: row.name,
        description: row.description,
        chunk: row.chunk,
        source: row.source,
        score: cosine(query.vector, row.embedding),
      }))
      .filter((hit) => hit.score >= minScore)
      .sort(
        (left, right) =>
          right.score - left.score ||
          left.name.localeCompare(right.name) ||
          left.chunk - right.chunk,
      )
      .slice(0, query.limit);
  }

  private async mutate(work: () => Promise<void>): Promise<void> {
    if (this.adapter.transaction) {
      await this.adapter.transaction(() => work());
      return;
    }
    await work();
  }
}

function storageRecord(record: SkillSearchChunkRecord): SkillSearchStorageRecord {
  return {
    id: record.id,
    kind: KIND_CHUNK,
    name: record.name,
    description: record.description,
    chunk: record.chunk,
    source: record.source,
    namespace: record.namespace,
    documentHash: record.documentHash,
    embedding: Array.from(record.embedding),
  };
}

function readMetadata(metadata: Readonly<Record<string, unknown>>): SkillVectorIndexMetadata {
  if (typeof metadata.indexVersion !== 'string' || typeof metadata.catalogVersion !== 'string') {
    throw new SkillAdapterError('INVALID_RESPONSE', 'Skill index metadata is corrupt');
  }
  return {
    indexVersion: metadata.indexVersion,
    catalogVersion: metadata.catalogVersion,
    ready: metadata.ready === true,
    dimensions: Number(metadata.dimensions),
    model: optionalString(metadata.model),
    revision: optionalString(metadata.revision),
    embedderId: optionalString(metadata.embedderId),
    namespace: optionalString(metadata.namespace),
    scoring: String(metadata.scoring ?? ''),
  };
}

function optionalString(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function zeroEmbedding(dimensions: number): number[] {
  const size = Number.isInteger(dimensions) && dimensions > 0 ? dimensions : 1;
  return Array.from({ length: size }, () => 0);
}

function groupByNamespace(
  records: readonly SkillSearchChunkRecord[],
): Map<string, SkillSearchChunkRecord[]> {
  const groups = new Map<string, SkillSearchChunkRecord[]>();
  for (const record of records) {
    const current = groups.get(record.namespace) ?? [];
    current.push(record);
    groups.set(record.namespace, current);
  }
  return groups;
}

function cosine(left: ArrayLike<number>, right: ArrayLike<number>): number {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index++) {
    const a = Number(left[index]);
    const b = Number(right[index]);
    dot += a * b;
    leftNorm += a * a;
    rightNorm += b * b;
  }
  return leftNorm === 0 || rightNorm === 0 ? 0 : dot / Math.sqrt(leftNorm * rightNorm);
}
