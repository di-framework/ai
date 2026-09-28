import type { Document } from '../document/document.ts';

/**
 * Embedding model contract (simplified vs Spring AI full Model call path).
 * Spring AI: {@code EmbeddingModel}.
 *
 * Provider packages implement real remote models; tests use {@code FakeEmbeddingModel}.
 */
export interface EmbeddingModel {
  /** Embed a single text string. */
  embed(text: string): number[] | Promise<number[]>;

  /** Embed document text content. */
  embedDocument(document: Document): number[] | Promise<number[]>;

  /** Batch embed texts. */
  embedBatch?(texts: readonly string[]): number[][] | Promise<number[][]>;

  /** Embedding dimensionality when known. */
  readonly dimensions?: number;
}

/**
 * Default document embedding uses {@link Document.text}.
 */
export async function embedDocument(model: EmbeddingModel, doc: Document): Promise<number[]> {
  if (typeof model.embedDocument === 'function') {
    return model.embedDocument(doc);
  }
  return model.embed(doc.text ?? '');
}
