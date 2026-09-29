import type { Message } from '../chat/messages/message.ts';

/**
 * Query in a RAG pipeline.
 * Spring AI: {@code org.springframework.ai.rag.Query}.
 */
export interface Query {
  readonly text: string;
  readonly history: readonly Message[];
  readonly context: Readonly<Record<string, unknown>>;
}

export interface QueryOptions {
  readonly text: string;
  readonly history?: readonly Message[];
  readonly context?: Readonly<Record<string, unknown>>;
}

export function query(options: QueryOptions | string): Query {
  if (typeof options === 'string') {
    return { text: options, history: [], context: {} };
  }
  if (options.text == null) {
    throw new Error('text cannot be null');
  }
  return {
    text: options.text,
    history: options.history ?? [],
    context: options.context ?? {},
  };
}

export function mutateQuery(q: Query, overrides: Partial<QueryOptions>): Query {
  return query({
    text: overrides.text ?? q.text,
    history: overrides.history ?? q.history,
    context: overrides.context ?? q.context,
  });
}
