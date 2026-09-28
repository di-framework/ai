import type { Document } from '../document/document.ts';
import type { EmbeddingModel } from '../embedding/embedding-model.ts';
import { l2Normalize } from '../embedding/fake-embedding-model.ts';
import type { SearchRequest } from './search-request.ts';

export async function resolveDocumentEmbedding(
  model: EmbeddingModel,
  document: Document,
): Promise<number[]> {
  const values =
    document.embedding != null
      ? Array.from(document.embedding)
      : [...(await model.embedDocument(document))];
  assertFiniteVector(values);
  if (model.dimensions && values.length !== model.dimensions) {
    throw new Error('Embedding dimension mismatch');
  }
  return unitVector(values);
}

export async function resolveQueryEmbedding(
  model: EmbeddingModel,
  request: SearchRequest,
): Promise<number[]> {
  const values =
    request.queryEmbedding != null
      ? Array.from(request.queryEmbedding)
      : [...(await model.embed(request.query))];
  assertFiniteVector(values);
  if (model.dimensions && values.length !== model.dimensions) {
    throw new Error('Embedding dimension mismatch');
  }
  return unitVector(values);
}

export function unitVector(values: number[]): number[] {
  let norm = 0;
  for (const value of values) norm += value * value;
  if (norm === 0) return values;
  if (Math.abs(Math.sqrt(norm) - 1) < 1e-5) return values;
  return l2Normalize(values);
}

export function assertFiniteVector(values: ArrayLike<number>): void {
  for (let index = 0; index < values.length; index++) {
    if (!Number.isFinite(Number(values[index]))) {
      throw new Error('Embedding contains a non-finite value');
    }
  }
}
