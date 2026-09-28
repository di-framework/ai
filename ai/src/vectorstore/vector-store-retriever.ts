import type { Document } from '../document/document.ts';
import { type SearchRequest, searchRequest } from './search-request.ts';

/**
 * Read-only vector similarity search.
 * Spring AI: {@code VectorStoreRetriever}.
 */
export interface VectorStoreRetriever {
  similaritySearch(request: SearchRequest): Promise<readonly Document[]>;

  /**
   * Convenience: search with default request settings for the query text.
   */
  similaritySearchQuery?(query: string): Promise<readonly Document[]>;
}

/** Default helper matching Spring's default method. */
export async function similaritySearchQuery(
  retriever: VectorStoreRetriever,
  query: string,
): Promise<readonly Document[]> {
  if (retriever.similaritySearchQuery) {
    return retriever.similaritySearchQuery(query);
  }
  return retriever.similaritySearch(searchRequest({ query }));
}
