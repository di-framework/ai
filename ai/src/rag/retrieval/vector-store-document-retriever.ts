import type { Document } from '../../document/document.ts';
import {
  type FilterExpression,
  parseFilterExpression,
  searchRequest,
  type VectorStore,
} from '../../vectorstore/index.ts';
import type { Query } from '../query.ts';
import type { DocumentRetriever } from './document-retriever.ts';

/**
 * Context key for a per-request filter expression (string or FilterExpression).
 * Spring AI: {@code VectorStoreDocumentRetriever.FILTER_EXPRESSION}.
 */
export const VECTOR_STORE_FILTER_EXPRESSION = 'vector_store_filter_expression';

export interface VectorStoreDocumentRetrieverOptions {
  readonly vectorStore: VectorStore;
  readonly similarityThreshold?: number;
  readonly topK?: number;
  readonly filterExpression?: FilterExpression | (() => FilterExpression | null);
}

/**
 * Retrieves documents via vector similarity search.
 * Spring AI: {@code VectorStoreDocumentRetriever}.
 */
export class VectorStoreDocumentRetriever implements DocumentRetriever {
  private readonly vectorStore: VectorStore;
  private readonly similarityThreshold: number;
  private readonly topK: number;
  private readonly filterExpression: () => FilterExpression | null;

  constructor(options: VectorStoreDocumentRetrieverOptions) {
    if (options.vectorStore == null) {
      throw new Error('vectorStore cannot be null');
    }
    if (options.similarityThreshold != null && options.similarityThreshold < 0) {
      throw new Error('similarityThreshold must be equal to or greater than 0.0');
    }
    if (options.topK != null && options.topK <= 0) {
      throw new Error('topK must be greater than 0');
    }
    this.vectorStore = options.vectorStore;
    this.similarityThreshold = options.similarityThreshold ?? 0;
    this.topK = options.topK ?? 4;
    if (typeof options.filterExpression === 'function') {
      this.filterExpression = options.filterExpression;
    } else if (options.filterExpression != null) {
      const fixed = options.filterExpression;
      this.filterExpression = () => fixed;
    } else {
      this.filterExpression = () => null;
    }
  }

  static builder(options: VectorStoreDocumentRetrieverOptions): VectorStoreDocumentRetriever {
    return new VectorStoreDocumentRetriever(options);
  }

  async retrieve(q: Query): Promise<readonly Document[]> {
    if (q == null) throw new Error('query cannot be null');
    const requestFilter = computeRequestFilterExpression(q, this.filterExpression);
    const request = searchRequest({
      query: q.text,
      filterExpression: requestFilter,
      similarityThreshold: this.similarityThreshold,
      topK: this.topK,
    });
    return this.vectorStore.similaritySearch(request);
  }
}

function computeRequestFilterExpression(
  q: Query,
  defaultFilter: () => FilterExpression | null,
): FilterExpression | null {
  const contextFilter = q.context[VECTOR_STORE_FILTER_EXPRESSION];
  if (contextFilter != null) {
    if (
      typeof contextFilter === 'object' &&
      contextFilter !== null &&
      'kind' in contextFilter &&
      (contextFilter as { kind: string }).kind === 'expression'
    ) {
      return contextFilter as FilterExpression;
    }
    const text = String(contextFilter);
    if (text.trim()) {
      return parseFilterExpression(text);
    }
  }
  return defaultFilter();
}
