import type { Document } from '../document/document.ts';
import type { FilterExpression } from './filter/index.ts';
import type { VectorStoreRetriever } from './vector-store-retriever.ts';

/**
 * Mutable vector store: write + retrieve.
 * Spring AI: {@code VectorStore} (extends DocumentWriter + VectorStoreRetriever).
 */
export interface VectorStore extends VectorStoreRetriever {
  /** Human-readable name (defaults to class name in Spring). */
  readonly name?: string;

  /** Add / upsert documents (embed + store). */
  add(documents: readonly Document[]): Promise<void>;

  /** Delete by document ids. */
  delete(ids: readonly string[]): Promise<void>;

  /** Delete documents matching a filter expression. */
  deleteByFilter?(filterExpression: FilterExpression): Promise<void>;

  /** Optional point lookup used by skill index metadata. */
  get?(id: string): Promise<Document | null>;
}
