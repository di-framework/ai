import type { Document } from '../document/document.ts';
import type { EmbeddingModel } from './embedding-model.ts';

/**
 * Embedding model for stores that receive already-computed vectors on documents
 * and {@code SearchRequest.queryEmbedding}.
 */
export class PrecomputedEmbeddingModel implements EmbeddingModel {
  constructor(readonly dimensions?: number) {}

  embed(_text: string): number[] {
    throw new Error('PrecomputedEmbeddingModel requires SearchRequest.queryEmbedding');
  }

  embedDocument(document: Document): number[] {
    if (document.embedding == null) {
      throw new Error('PrecomputedEmbeddingModel requires Document.embedding');
    }
    const values = Array.from(document.embedding);
    if (this.dimensions != null && values.length !== this.dimensions) {
      throw new Error('Embedding dimension mismatch');
    }
    return values;
  }
}
