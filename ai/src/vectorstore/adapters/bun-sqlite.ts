import type { Document } from '../../document/document.ts';
import { withDocumentScore } from '../../document/document.ts';
import type { EmbeddingModel } from '../../embedding/embedding-model.ts';
import { evaluateFilterExpression, type FilterExpression } from '../filter/index.ts';
import { resolveDocumentEmbedding, resolveQueryEmbedding } from '../resolve-embedding.ts';
import { type SearchRequest, searchRequest } from '../search-request.ts';
import type { VectorStore } from '../vector-store.ts';
import { cosine, HnswIndex, type HnswSnapshot } from './hnsw.ts';
import { loadWasmSimilarity, type WasmSimilarity } from './wasm-similarity-loader.ts';

export interface BunVectorDatabase {
  run(sql: string, ...args: unknown[]): { changes?: number };
  query(sql: string): { all(...args: unknown[]): unknown[] };
}

export type VectorSearchMode = 'auto' | 'exact' | 'ann';

export interface BunSqliteVectorStoreOptions {
  db: BunVectorDatabase;
  embeddingModel: EmbeddingModel;
  table?: string;
  name?: string;
  searchMode?: VectorSearchMode;
  exactScanLimit?: number;
  hnsw?: {
    readonly M?: number;
    readonly efConstruction?: number;
    readonly efSearch?: number;
  };
}

const FORMAT_VERSION = 'bun-sqlite-vector/v2';
const DEFAULT_EXACT_SCAN_LIMIT = 8192;
interface VectorRow {
  readonly id: string;
  readonly text: string;
  readonly metadata: Record<string, unknown>;
  readonly embedding: number[];
}

interface ScoredRow {
  readonly r: VectorRow;
  readonly score: number;
}

let wasmPromise: Promise<WasmSimilarity | null> | undefined;

function loadWasm(): Promise<WasmSimilarity | null> {
  wasmPromise ??= loadWasmSimilarity();
  return wasmPromise;
}

export function rankCosineScores(
  query: number[],
  candidates: readonly VectorRow[],
  wasm: WasmSimilarity | null,
): ScoredRow[] {
  if (!wasm) return candidates.map((r) => ({ r, score: cosine(query, r.embedding) }));
  const pairs = wasm.cosine_similarity_dataspace(
    new Float64Array(candidates.flatMap((r) => r.embedding)),
    candidates.length,
    query.length,
    new Float64Array(query),
  );
  const scored: ScoredRow[] = [];
  for (let i = 0; i + 1 < pairs.length; i += 2) {
    const candidate = candidates[pairs[i + 1] as number];
    if (candidate) scored.push({ r: candidate, score: pairs[i] as number });
  }
  return scored;
}

export class BunSqliteVectorStore implements VectorStore {
  readonly name: string;
  readonly searchMode: VectorSearchMode;
  readonly exactScanLimit: number;
  private readonly table: string;
  private readonly metaTable: string;
  private readonly graphTable: string;
  private readonly model: EmbeddingModel;
  private readonly db: BunVectorDatabase;
  private readonly hnswOptions: NonNullable<BunSqliteVectorStoreOptions['hnsw']>;
  private cache:
    | {
        generation: string;
        rows: VectorRow[];
        graph: HnswIndex | null;
      }
    | undefined;

  constructor(options: BunSqliteVectorStoreOptions) {
    this.db = options.db;
    this.model = options.embeddingModel;
    this.table = options.table ?? 'ai_vectors';
    this.metaTable = `${this.table}_store_meta`;
    this.graphTable = `${this.table}_hnsw`;
    this.name = options.name ?? 'BunSqliteVectorStore';
    this.searchMode = options.searchMode ?? 'auto';
    this.exactScanLimit = options.exactScanLimit ?? DEFAULT_EXACT_SCAN_LIMIT;
    this.hnswOptions = options.hnsw ?? {};
    if (!Number.isInteger(this.exactScanLimit) || this.exactScanLimit < 1) {
      throw new Error('exactScanLimit must be a positive integer');
    }
    this.db.run(
      `CREATE TABLE IF NOT EXISTS "${this.table}" (id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata TEXT NOT NULL, embedding BLOB NOT NULL)`,
    );
    this.db.run(
      `CREATE TABLE IF NOT EXISTS "${this.metaTable}" (key TEXT PRIMARY KEY, value TEXT NOT NULL)`,
    );
    this.db.run(
      `CREATE TABLE IF NOT EXISTS "${this.graphTable}" (id TEXT PRIMARY KEY, level INTEGER NOT NULL, neighbors TEXT NOT NULL)`,
    );
    const format = this.meta('formatVersion');
    if (format != null && format !== FORMAT_VERSION && format !== '1') {
      throw new Error(`Unsupported vector store format '${format}'`);
    }
    if (format == null) this.setMeta('formatVersion', FORMAT_VERSION);
  }

  async add(documents: readonly Document[]): Promise<void> {
    for (const doc of documents) {
      const vector = await resolveDocumentEmbedding(this.model, doc);
      this.db.run(
        `INSERT INTO "${this.table}" (id,text,metadata,embedding) VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET text=excluded.text, metadata=excluded.metadata, embedding=excluded.embedding`,
        doc.id,
        doc.text ?? '',
        JSON.stringify(doc.metadata),
        packing(vector),
      );
    }
    this.bumpDataGeneration();
    this.syncGraph(this.rows());
  }

  async get(id: string): Promise<Document | null> {
    const row = this.db
      .query(`SELECT id,text,metadata,embedding FROM "${this.table}" WHERE id = ?`)
      .all(id)[0];
    if (row == null) return null;
    const decoded = decodeRow(row);
    return {
      id: decoded.id,
      text: decoded.text,
      media: null,
      metadata: decoded.metadata,
      score: null,
      embedding: decoded.embedding,
    };
  }

  async delete(ids: readonly string[]): Promise<void> {
    for (const id of ids) this.db.run(`DELETE FROM "${this.table}" WHERE id = ?`, id);
    this.bumpDataGeneration();
    this.syncGraph(this.rows());
  }

  async deleteByFilter(filter: FilterExpression): Promise<void> {
    const rows = this.rows().filter((r) => evaluateFilterExpression(filter, r.metadata));
    await this.delete(rows.map((r) => r.id));
  }

  async similaritySearch(request: SearchRequest): Promise<readonly Document[]> {
    const query = await resolveQueryEmbedding(this.model, request);
    const candidates = this.rows().filter(
      (r) =>
        !request.filterExpression || evaluateFilterExpression(request.filterExpression, r.metadata),
    );
    if (candidates.length === 0) return [];
    const ranked = (await this.rank(query, candidates, request)).sort((a, b) => b.score - a.score);
    return ranked
      .filter((x) => x.score >= request.similarityThreshold)
      .slice(0, request.topK)
      .map(({ r, score }) =>
        withDocumentScore(
          { id: r.id, text: r.text, media: null, metadata: r.metadata, score: null },
          score,
        ),
      );
  }

  similaritySearchQuery(query: string) {
    return this.similaritySearch(searchRequest({ query }));
  }

  private async rank(
    query: number[],
    candidates: readonly VectorRow[],
    request: SearchRequest,
  ): Promise<ScoredRow[]> {
    const useAnn = this.shouldUseAnn(this.rows().length);
    if (!useAnn) return this.scoreExact(query, candidates);
    const graph = this.requireGraph(this.rows());
    const overFetch = request.filterExpression
      ? Math.max(request.topK * 8, request.topK)
      : request.topK;
    const hits = graph.search(query, Math.min(overFetch, candidates.length));
    const byId = new Map(candidates.map((row) => [row.id, row]));
    const scored: ScoredRow[] = [];
    for (const hit of hits) {
      const row = byId.get(hit.id);
      if (row) scored.push({ r: row, score: hit.score });
    }
    return scored;
  }

  private async scoreExact(
    query: number[],
    candidates: readonly VectorRow[],
  ): Promise<ScoredRow[]> {
    return rankCosineScores(query, candidates, await loadWasm());
  }

  private shouldUseAnn(count: number): boolean {
    if (this.searchMode === 'exact') return false;
    if (this.searchMode === 'ann') return true;
    return count > this.exactScanLimit;
  }

  private requireGraph(rows: readonly VectorRow[]): HnswIndex {
    const generation = this.meta('dataGeneration') ?? '0';
    if (this.cache?.generation === generation && this.cache.graph) return this.cache.graph;
    const graphGeneration = this.meta('graphGeneration');
    if (graphGeneration !== generation || this.meta('graphReady') !== 'true') {
      if (this.searchMode === 'ann') {
        throw new Error('ANN graph is not ready');
      }
      return this.syncGraph(rows) ?? this.buildGraph(rows);
    }
    const restored = this.loadGraph(rows);
    if (!restored) {
      if (this.searchMode === 'ann') throw new Error('ANN graph is not ready');
      return this.syncGraph(rows) ?? this.buildGraph(rows);
    }
    this.cache = { generation, rows: [...rows], graph: restored };
    return restored;
  }

  private syncGraph(rows: readonly VectorRow[]): HnswIndex | null {
    const generation = this.meta('dataGeneration') ?? '0';
    if (!this.shouldUseAnn(rows.length)) {
      this.db.run(`DELETE FROM "${this.graphTable}"`);
      this.setMeta('graphGeneration', generation);
      this.setMeta('graphReady', 'true');
      this.cache = { generation, rows: [...rows], graph: null };
      return null;
    }
    const graph = this.buildGraph(rows);
    this.persistGraph(graph);
    this.setMeta('graphGeneration', generation);
    this.setMeta('graphReady', 'true');
    this.cache = { generation, rows: [...rows], graph };
    return graph;
  }

  private buildGraph(rows: readonly VectorRow[]): HnswIndex {
    const graph = new HnswIndex(this.hnswOptions);
    for (const row of [...rows].sort((left, right) => left.id.localeCompare(right.id))) {
      graph.insert(row.id, row.embedding);
    }
    return graph;
  }

  private persistGraph(graph: HnswIndex): void {
    this.db.run(`DELETE FROM "${this.graphTable}"`);
    const snapshot = graph.snapshot();
    this.setMeta('graphEntryPoint', snapshot.entryPoint ?? '');
    this.setMeta('graphM', String(snapshot.M));
    this.setMeta('graphEfConstruction', String(snapshot.efConstruction));
    this.setMeta('graphEfSearch', String(snapshot.efSearch));
    for (const node of snapshot.nodes) {
      this.db.run(
        `INSERT INTO "${this.graphTable}" (id, level, neighbors) VALUES (?,?,?)`,
        node.id,
        node.level,
        JSON.stringify(node.neighbors),
      );
    }
  }

  private loadGraph(rows: readonly VectorRow[]): HnswIndex | null {
    const stored = this.db.query(`SELECT id, level, neighbors FROM "${this.graphTable}"`).all();
    if (stored.length === 0) return null;
    const embeddings = new Map(rows.map((row) => [row.id, Float32Array.from(row.embedding)]));
    const snapshot: HnswSnapshot = {
      M: Number(this.meta('graphM') ?? this.hnswOptions.M ?? 16),
      efConstruction: Number(
        this.meta('graphEfConstruction') ?? this.hnswOptions.efConstruction ?? 200,
      ),
      efSearch: Number(this.meta('graphEfSearch') ?? this.hnswOptions.efSearch ?? 64),
      entryPoint: this.meta('graphEntryPoint') || null,
      nodes: stored.map((item) => {
        const row = item as { id: unknown; level: unknown; neighbors: unknown };
        let neighbors: string[][] = [];
        try {
          neighbors = JSON.parse(String(row.neighbors)) as string[][];
        } catch {
          throw new Error('ANN graph is corrupt');
        }
        if (!Array.isArray(neighbors)) throw new Error('ANN graph is corrupt');
        return { id: String(row.id), level: Number(row.level), neighbors };
      }),
    };
    return HnswIndex.restore(snapshot, embeddings, this.hnswOptions);
  }

  private rows(): VectorRow[] {
    const generation = this.meta('dataGeneration') ?? '0';
    if (this.cache?.generation === generation) return this.cache.rows;
    const rows = this.db
      .query(`SELECT id,text,metadata,embedding FROM "${this.table}"`)
      .all()
      .map((x) => decodeRow(x));
    this.cache = {
      generation,
      rows,
      graph: this.cache?.generation === generation ? this.cache.graph : null,
    };
    return rows;
  }

  private bumpDataGeneration(): void {
    const next = String(Number(this.meta('dataGeneration') ?? '0') + 1);
    this.setMeta('dataGeneration', next);
    this.setMeta('formatVersion', FORMAT_VERSION);
    this.setMeta('graphReady', 'false');
    this.cache = undefined;
  }

  private meta(key: string): string | undefined {
    const row = this.db.query(`SELECT value FROM "${this.metaTable}" WHERE key = ?`).all(key)[0] as
      | { value?: unknown }
      | undefined;
    return row?.value == null ? undefined : String(row.value);
  }

  private setMeta(key: string, value: string): void {
    this.db.run(
      `INSERT INTO "${this.metaTable}" (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
      key,
      value,
    );
  }
}

function packing(vector: number[]): Uint8Array {
  return new Uint8Array(Float32Array.from(vector).buffer);
}

function decodeRow(x: unknown): VectorRow {
  const r = x as { id: unknown; text: unknown; metadata: unknown; embedding: unknown };
  let metadata: Record<string, unknown>;
  try {
    metadata = JSON.parse(String(r.metadata));
  } catch {
    throw new Error('Vector metadata is corrupt');
  }
  return {
    id: String(r.id),
    text: String(r.text),
    metadata,
    embedding: decodeEmbedding(r.embedding),
  };
}

function decodeEmbedding(value: unknown): number[] {
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value) as unknown;
      if (!Array.isArray(parsed)) throw new Error('Embedding dimension mismatch');
      return parsed.map((item) => Number(item));
    } catch (error) {
      if (error instanceof Error && error.message === 'Embedding dimension mismatch') throw error;
      throw new Error('Vector embedding is corrupt');
    }
  }
  const bytes = toBytes(value);
  if (bytes.byteLength % 4 !== 0) throw new Error('Vector embedding is corrupt');
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const floats = new Float32Array(copy.buffer);
  const values: number[] = [];
  for (let index = 0; index < floats.length; index++) {
    const item = floats[index];
    if (!Number.isFinite(item)) throw new Error('Vector embedding is corrupt');
    values.push(item as number);
  }
  return values;
}

function toBytes(value: unknown): Uint8Array {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  throw new Error('Vector embedding is corrupt');
}
