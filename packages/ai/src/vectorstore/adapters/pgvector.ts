import type { Document } from '../../document/document.ts';
import { withDocumentScore } from '../../document/document.ts';
import type { EmbeddingModel } from '../../embedding/embedding-model.ts';
import { resolveDocumentEmbedding, resolveQueryEmbedding } from '../resolve-embedding.ts';
import { type SearchRequest, searchRequest } from '../search-request.ts';
import type { VectorStore } from '../vector-store.ts';
export interface PgClient {
  query<T = any>(sql: string, params?: unknown[]): Promise<{ rows: T[] }> | { rows: T[] };
}
export interface PgVectorStoreOptions {
  client: PgClient;
  embeddingModel: EmbeddingModel;
  table?: string;
  name?: string;
}
export class PgVectorStore implements VectorStore {
  readonly name: string;
  private readonly c: PgClient;
  private readonly model: EmbeddingModel;
  private readonly table: string;
  constructor(options: PgVectorStoreOptions) {
    this.c = options.client;
    this.model = options.embeddingModel;
    this.table = options.table ?? 'ai_vectors';
    this.name = options.name ?? 'PgVectorStore';
  }
  async add(documents: readonly Document[]) {
    for (const d of documents) {
      const v = await resolveDocumentEmbedding(this.model, d);
      await this.c.query(
        `INSERT INTO ${this.table} (id, content, metadata, embedding) VALUES ($1,$2,$3,$4) ON CONFLICT (id) DO UPDATE SET content=$2, metadata=$3, embedding=$4`,
        [d.id, d.text ?? '', JSON.stringify(d.metadata), `[${v.join(',')}]`],
      );
    }
  }
  async get(id: string) {
    const result = await this.c.query<{ id: string; content: string; metadata: unknown }>(
      `SELECT id, content, metadata FROM ${this.table} WHERE id=$1`,
      [id],
    );
    const row = result.rows[0];
    if (!row) return null;
    return {
      id: row.id,
      text: row.content,
      media: null,
      metadata: typeof row.metadata === 'string' ? JSON.parse(row.metadata) : (row.metadata ?? {}),
      score: null,
    };
  }

  async delete(ids: readonly string[]) {
    for (const id of ids) await this.c.query(`DELETE FROM ${this.table} WHERE id=$1`, [id]);
  }
  async similaritySearch(request: SearchRequest) {
    const v = await resolveQueryEmbedding(this.model, request);
    const result = await this.c.query<any>(
      `SELECT id, content, metadata, 1 - (embedding <=> $1::vector) AS score FROM ${this.table} ORDER BY embedding <=> $1::vector LIMIT $2`,
      [`[${v.join(',')}]`, request.topK],
    );
    return result.rows
      .filter((r) => Number(r.score) >= request.similarityThreshold)
      .map((r) =>
        withDocumentScore(
          {
            id: r.id,
            text: r.content,
            media: null,
            metadata: typeof r.metadata === 'string' ? JSON.parse(r.metadata) : (r.metadata ?? {}),
            score: null,
          },
          Number(r.score),
        ),
      );
  }
  similaritySearchQuery(query: string) {
    return this.similaritySearch(searchRequest({ query }));
  }
}
