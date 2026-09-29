import { type FilterExpression, parseFilterExpression } from './filter/index.ts';

/** Accept all scores (disable threshold filtering). Spring AI constant. */
export const SIMILARITY_THRESHOLD_ACCEPT_ALL = 0.0;

/** Default top-k. Spring AI: {@code SearchRequest.DEFAULT_TOP_K}. */
export const DEFAULT_TOP_K = 4;

export interface SearchRequest {
  readonly query: string;
  readonly topK: number;
  readonly similarityThreshold: number;
  readonly filterExpression: FilterExpression | null;
  /** When set, stores skip {@code EmbeddingModel.embed} and search this vector. */
  readonly queryEmbedding?: ArrayLike<number>;
}

export interface SearchRequestOptions {
  readonly query?: string;
  readonly topK?: number;
  readonly similarityThreshold?: number;
  readonly filterExpression?: FilterExpression | string | null;
  readonly queryEmbedding?: ArrayLike<number>;
}

/**
 * Similarity search request.
 * Spring AI: {@code SearchRequest}.
 */
export function searchRequest(options: SearchRequestOptions = {}): SearchRequest {
  const topK = options.topK ?? DEFAULT_TOP_K;
  if (topK <= 0) {
    throw new Error('TopK should be positive.');
  }
  const similarityThreshold = options.similarityThreshold ?? SIMILARITY_THRESHOLD_ACCEPT_ALL;
  if (similarityThreshold < 0 || similarityThreshold > 1) {
    throw new Error('Similarity threshold must be in [0,1] range.');
  }

  let filterExpression: FilterExpression | null = null;
  if (typeof options.filterExpression === 'string') {
    filterExpression = parseFilterExpression(options.filterExpression);
  } else if (options.filterExpression != null) {
    filterExpression = options.filterExpression;
  }

  return {
    query: options.query ?? '',
    topK,
    similarityThreshold,
    filterExpression,
    ...(options.queryEmbedding != null ? { queryEmbedding: options.queryEmbedding } : {}),
  };
}

export class SearchRequestBuilder {
  private options: SearchRequestOptions = {};

  query(query: string): this {
    if (query == null) throw new Error('Query can not be null.');
    this.options = { ...this.options, query };
    return this;
  }

  topK(topK: number): this {
    this.options = { ...this.options, topK };
    return this;
  }

  similarityThreshold(threshold: number): this {
    this.options = { ...this.options, similarityThreshold: threshold };
    return this;
  }

  similarityThresholdAll(): this {
    this.options = {
      ...this.options,
      similarityThreshold: SIMILARITY_THRESHOLD_ACCEPT_ALL,
    };
    return this;
  }

  filterExpression(expression: FilterExpression | string | null | undefined): this {
    this.options = {
      ...this.options,
      filterExpression: expression ?? null,
    };
    return this;
  }

  queryEmbedding(embedding: ArrayLike<number>): this {
    this.options = { ...this.options, queryEmbedding: embedding };
    return this;
  }

  build(): SearchRequest {
    return searchRequest(this.options);
  }
}

export function searchRequestBuilder(): SearchRequestBuilder {
  return new SearchRequestBuilder();
}
