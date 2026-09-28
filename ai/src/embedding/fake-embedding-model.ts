import type { Document } from '../document/document.ts';
import type { EmbeddingModel } from './embedding-model.ts';

export interface FakeEmbeddingModelOptions {
  /** Vector size. Default 64. */
  readonly dimensions?: number;
}

/**
 * Deterministic bag-of-words embedding for tests and SimpleVectorStore demos.
 * Similar token overlap → higher cosine similarity.
 */
export class FakeEmbeddingModel implements EmbeddingModel {
  readonly dimensions: number;

  constructor(options: FakeEmbeddingModelOptions = {}) {
    this.dimensions = options.dimensions ?? 64;
    if (this.dimensions <= 0) {
      throw new Error('dimensions must be greater than 0');
    }
  }

  embed(text: string): number[] {
    return bagOfWordsEmbedding(text ?? '', this.dimensions);
  }

  embedDocument(document: Document): number[] {
    return this.embed(document.text ?? '');
  }

  embedBatch(texts: readonly string[]): number[][] {
    return texts.map((t) => this.embed(t));
  }
}

/** Hash tokens into a fixed-dim dense vector and L2-normalize. */
export function bagOfWordsEmbedding(text: string, dimensions = 64): number[] {
  const vec = new Array<number>(dimensions).fill(0);
  const tokens = text.toLowerCase().split(/\W+/).filter(Boolean);
  for (const token of tokens) {
    let h = 0;
    for (let i = 0; i < token.length; i++) {
      h = (Math.imul(h, 31) + token.charCodeAt(i)) | 0;
    }
    const idx = Math.abs(h) % dimensions;
    vec[idx]! += 1;
    // Bigram-ish second feature for short phrases
    if (token.length > 1) {
      const idx2 = Math.abs(h >>> 1) % dimensions;
      vec[idx2]! += 0.5;
    }
  }
  return l2Normalize(vec);
}

export function l2Normalize(vec: readonly number[]): number[] {
  let sum = 0;
  for (const v of vec) sum += v * v;
  const norm = Math.sqrt(sum);
  if (norm === 0) return [...vec];
  return vec.map((v) => v / norm);
}

/** Cosine similarity of two vectors (assumes same length). */
export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) {
    throw new Error(`embedding dimension mismatch: ${a.length} vs ${b.length}`);
  }
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = b[i]!;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  if (denom === 0) return 0;
  // Clamp numerical noise into [-1, 1]
  const sim = dot / denom;
  return Math.max(-1, Math.min(1, sim));
}
