import type { Document } from '../../document/document.ts';
import type { Query } from '../query.ts';

/**
 * Joins documents retrieved for multiple queries / sources.
 * Spring AI: {@code DocumentJoiner}.
 */
export interface DocumentJoiner {
  join(
    documentsForQuery: ReadonlyMap<Query, readonly (readonly Document[])[]>,
  ): readonly Document[];
}

/**
 * Concatenate and de-duplicate by document id (first wins).
 * Spring AI: {@code ConcatenationDocumentJoiner}.
 */
export class ConcatenationDocumentJoiner implements DocumentJoiner {
  join(
    documentsForQuery: ReadonlyMap<Query, readonly (readonly Document[])[]>,
  ): readonly Document[] {
    const seen = new Set<string>();
    const result: Document[] = [];
    for (const lists of documentsForQuery.values()) {
      for (const list of lists) {
        for (const doc of list) {
          if (seen.has(doc.id)) continue;
          seen.add(doc.id);
          result.push(doc);
        }
      }
    }
    return result;
  }
}
