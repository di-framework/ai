import type { Document } from '../document/document.ts';
import { withDocumentScore } from '../document/document.ts';
import type { EmbeddingModel } from '../embedding/embedding-model.ts';
import { cosineSimilarity } from '../embedding/fake-embedding-model.ts';
import { evaluateFilterExpression, type FilterExpression } from './filter/index.ts';
import { resolveDocumentEmbedding, resolveQueryEmbedding } from './resolve-embedding.ts';
import { type SearchRequest, searchRequest } from './search-request.ts';
import type { VectorStore } from './vector-store.ts';

interface StoredEntry {
  readonly id: string;
  readonly text: string;
  readonly metadata: Readonly<Record<string, unknown>>;
  readonly embedding: number[];
}

export interface SimpleVectorStoreOptions {
  readonly embeddingModel: EmbeddingModel;
  readonly name?: string;
}

/**
 * In-memory vector store with cosine similarity and metadata filters.
 * Spring AI: {@code SimpleVectorStore} (test / demo use).
 */
export class SimpleVectorStore implements VectorStore {
  readonly name: string;
  private readonly embeddingModel: EmbeddingModel;
  private readonly store = new Map<string, StoredEntry>();

  constructor(options: SimpleVectorStoreOptions) {
    if (options.embeddingModel == null) {
      throw new Error('embeddingModel cannot be null');
    }
    this.embeddingModel = options.embeddingModel;
    this.name = options.name ?? 'SimpleVectorStore';
  }

  static builder(embeddingModel: EmbeddingModel): SimpleVectorStoreBuilder {
    return new SimpleVectorStoreBuilder(embeddingModel);
  }

  static of(embeddingModel: EmbeddingModel): SimpleVectorStore {
    return new SimpleVectorStore({ embeddingModel });
  }

  /** Number of stored documents (test helper). */
  get size(): number {
    return this.store.size;
  }

  async add(documents: readonly Document[]): Promise<void> {
    if (documents == null) {
      throw new Error('Documents list cannot be null');
    }
    if (documents.length === 0) {
      throw new Error('Documents list cannot be empty');
    }
    for (const doc of documents) {
      const text = doc.text ?? '';
      const embedding = await resolveDocumentEmbedding(this.embeddingModel, doc);
      this.store.set(doc.id, {
        id: doc.id,
        text,
        metadata: { ...doc.metadata },
        embedding: [...embedding],
      });
    }
  }

  async get(id: string): Promise<Document | null> {
    const entry = this.store.get(id);
    if (!entry) return null;
    return {
      id: entry.id,
      text: entry.text,
      media: null,
      metadata: entry.metadata,
      score: null,
      embedding: entry.embedding,
    };
  }

  async delete(ids: readonly string[]): Promise<void> {
    for (const id of ids) {
      this.store.delete(id);
    }
  }

  async deleteByFilter(filterExpression: FilterExpression): Promise<void> {
    const toDelete: string[] = [];
    for (const entry of this.store.values()) {
      if (evaluateFilterExpression(filterExpression, entry.metadata)) {
        toDelete.push(entry.id);
      }
    }
    await this.delete(toDelete);
  }

  async similaritySearch(request: SearchRequest): Promise<readonly Document[]> {
    const queryEmbedding = await resolveQueryEmbedding(this.embeddingModel, request);
    const results: Document[] = [];

    for (const entry of this.store.values()) {
      if (
        request.filterExpression != null &&
        !evaluateFilterExpression(request.filterExpression, entry.metadata)
      ) {
        continue;
      }
      const score = cosineSimilarity(queryEmbedding, entry.embedding);
      if (score < request.similarityThreshold) {
        continue;
      }
      results.push(
        withDocumentScore(
          {
            id: entry.id,
            text: entry.text,
            media: null,
            metadata: entry.metadata,
            score: null,
          },
          score,
        ),
      );
    }

    results.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
    return results.slice(0, request.topK);
  }

  async similaritySearchQuery(query: string): Promise<readonly Document[]> {
    return this.similaritySearch(searchRequest({ query }));
  }
}

export class SimpleVectorStoreBuilder {
  private nameValue: string | undefined;

  constructor(private readonly embeddingModel: EmbeddingModel) {}

  name(name: string): this {
    this.nameValue = name;
    return this;
  }

  build(): SimpleVectorStore {
    return new SimpleVectorStore({
      embeddingModel: this.embeddingModel,
      name: this.nameValue,
    });
  }
}
