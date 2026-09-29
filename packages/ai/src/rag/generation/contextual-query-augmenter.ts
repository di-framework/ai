import { renderTemplate } from '../../chat/client/template.ts';
import type { Document } from '../../document/document.ts';
import { type Query, query } from '../query.ts';
import type { QueryAugmenter } from './query-augmenter.ts';

const DEFAULT_PROMPT_TEMPLATE = `Context information is below.

---------------------
{context}
---------------------

Given the context information and no prior knowledge, answer the query.

Follow these rules:

1. If the answer is not in the context, just say that you don't know.
2. Avoid statements like "Based on the context..." or "The provided information...".

Query: {query}

Answer:
`;

const DEFAULT_EMPTY_CONTEXT_PROMPT_TEMPLATE = `The user query is outside your knowledge base.
Politely inform the user that you can't answer it.
`;

export interface ContextualQueryAugmenterOptions {
  readonly promptTemplate?: string;
  readonly emptyContextPromptTemplate?: string;
  /** When true, empty retrieval returns the original query. Default false. */
  readonly allowEmptyContext?: boolean;
  readonly documentFormatter?: (documents: readonly Document[]) => string;
}

/**
 * Augments the user query with retrieved document context.
 * Spring AI: {@code ContextualQueryAugmenter}.
 */
export class ContextualQueryAugmenter implements QueryAugmenter {
  private readonly promptTemplate: string;
  private readonly emptyContextPromptTemplate: string;
  private readonly allowEmptyContext: boolean;
  private readonly documentFormatter: (documents: readonly Document[]) => string;

  constructor(options: ContextualQueryAugmenterOptions = {}) {
    this.promptTemplate = options.promptTemplate ?? DEFAULT_PROMPT_TEMPLATE;
    this.emptyContextPromptTemplate =
      options.emptyContextPromptTemplate ?? DEFAULT_EMPTY_CONTEXT_PROMPT_TEMPLATE;
    this.allowEmptyContext = options.allowEmptyContext ?? false;
    this.documentFormatter =
      options.documentFormatter ?? ((docs) => docs.map((d) => d.text ?? '').join('\n'));
  }

  static builder(options: ContextualQueryAugmenterOptions = {}): ContextualQueryAugmenter {
    return new ContextualQueryAugmenter(options);
  }

  augment(q: Query, documents: readonly Document[]): Query {
    if (q == null) throw new Error('query cannot be null');
    if (documents == null) throw new Error('documents cannot be null');

    if (documents.length === 0) {
      if (this.allowEmptyContext) {
        return q;
      }
      return query(renderTemplate(this.emptyContextPromptTemplate, {}));
    }

    const documentContext = this.documentFormatter(documents);
    const text = renderTemplate(this.promptTemplate, {
      query: q.text,
      context: documentContext,
    });
    return query({
      text,
      history: q.history,
      context: q.context,
    });
  }
}
