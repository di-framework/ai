import type { Document } from '../../document/document.ts';
import type { Query } from '../query.ts';

/**
 * Retrieves documents relevant to a {@link Query}.
 * Spring AI: {@code DocumentRetriever}.
 */
export interface DocumentRetriever {
  retrieve(query: Query): Promise<readonly Document[]>;
}
