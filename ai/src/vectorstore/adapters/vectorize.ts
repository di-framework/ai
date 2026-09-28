import type { Document } from '../../document/document.ts';
import { withDocumentScore } from '../../document/document.ts';
import type { EmbeddingModel } from '../../embedding/embedding-model.ts';
import { resolveDocumentEmbedding, resolveQueryEmbedding } from '../resolve-embedding.ts';
import { type SearchRequest, searchRequest } from '../search-request.ts';
import type { VectorStore } from '../vector-store.ts';
export interface VectorizeIndex {
  upsert(vectors: unknown[]): Promise<unknown>;
  query(
    vector: number[],
    options?: Record<string, unknown>,
  ): Promise<{
    matches?: Array<{ id: string; score?: number; metadata?: Record<string, unknown> }>;
  }>;
  deleteByIds(ids: string[]): Promise<unknown>;
}
export interface VectorizeVectorStoreOptions {
  index: VectorizeIndex;
  embeddingModel: EmbeddingModel;
  name?: string;
}
export class VectorizeVectorStore implements VectorStore {
  readonly name: string;
  private readonly index: VectorizeIndex;
  private readonly model: EmbeddingModel;
  private readonly docs = new Map<string, Document>();
  constructor(options: VectorizeVectorStoreOptions) {
    this.index = options.index;
    this.model = options.embeddingModel;
    this.name = options.name ?? 'VectorizeVectorStore';
  }
  async add(documents: readonly Document[]) {
    const vectors = [];
    for (const doc of documents) {
      const values = await resolveDocumentEmbedding(this.model, doc);
      vectors.push({ id: doc.id, values, metadata: { ...doc.metadata, text: doc.text ?? '' } });
      this.docs.set(doc.id, doc);
    }
    await this.index.upsert(vectors);
  }
  async get(id: string) {
    return this.docs.get(id) ?? null;
  }

  async delete(ids: readonly string[]) {
    await this.index.deleteByIds([...ids]);
    for (const id of ids) this.docs.delete(id);
  }
  async similaritySearch(request: SearchRequest) {
    const matches =
      (
        await this.index.query(await resolveQueryEmbedding(this.model, request), {
          topK: request.topK,
          returnMetadata: true,
        })
      ).matches ?? [];
    return matches
      .filter((m) => (m.score ?? 0) >= request.similarityThreshold)
      .map((m) => {
        const d = this.docs.get(m.id) ?? {
          id: m.id,
          text: String(m.metadata?.text ?? ''),
          media: null,
          metadata: m.metadata ?? {},
          score: null,
        };
        return withDocumentScore(d, m.score ?? 0);
      });
  }
  similaritySearchQuery(query: string) {
    return this.similaritySearch(searchRequest({ query }));
  }
}
