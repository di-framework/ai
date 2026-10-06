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

/**
 * A Vectorize index, the `@di-framework/cloudflare` `{ binding }` descriptor,
 * or a getter that reads the current Worker env.
 */
export type VectorizeSource = VectorizeIndex | { readonly binding?: unknown } | (() => unknown);

export interface VectorizeVectorStoreOptions {
  index: VectorizeSource;
  embeddingModel: EmbeddingModel;
  name?: string;
}

export function resolveVectorizeIndex(source: VectorizeSource): VectorizeIndex {
  const resolved = typeof source === 'function' ? source() : source;
  if (isVectorizeIndex(resolved)) return resolved;
  if (
    resolved &&
    typeof resolved === 'object' &&
    'binding' in resolved &&
    isVectorizeIndex(resolved.binding)
  ) {
    return resolved.binding;
  }
  throw new Error(
    'Vectorize index is not available. Publish the Worker env before using the store.',
  );
}

function isVectorizeIndex(value: unknown): value is VectorizeIndex {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as VectorizeIndex).query === 'function' &&
    typeof (value as VectorizeIndex).upsert === 'function'
  );
}

export class VectorizeVectorStore implements VectorStore {
  readonly name: string;
  private readonly source: VectorizeSource;
  private readonly model: EmbeddingModel;
  private readonly docs = new Map<string, Document>();
  constructor(options: VectorizeVectorStoreOptions) {
    this.source = options.index;
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
    await this.index().upsert(vectors);
  }
  async get(id: string) {
    return this.docs.get(id) ?? null;
  }

  async delete(ids: readonly string[]) {
    await this.index().deleteByIds([...ids]);
    for (const id of ids) this.docs.delete(id);
  }
  async similaritySearch(request: SearchRequest) {
    const matches =
      (
        await this.index().query(await resolveQueryEmbedding(this.model, request), {
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

  private index(): VectorizeIndex {
    return resolveVectorizeIndex(this.source);
  }
}
