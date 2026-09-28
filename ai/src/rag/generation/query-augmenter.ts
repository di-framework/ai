import type { Document } from '../../document/document.ts';
import type { Query } from '../query.ts';

/**
 * Augments a user query with retrieved document context.
 * Spring AI: {@code QueryAugmenter}.
 */
export interface QueryAugmenter {
  augment(query: Query, documents: readonly Document[]): Query;
}
