import type { Usage } from './usage.ts';

/**
 * Response-level metadata, aligned with Spring AI {@code ChatResponseMetadata}.
 */
export interface ChatResponseMetadata {
  readonly id?: string;
  readonly model?: string;
  readonly usage?: Usage;
  readonly rateLimit?: Readonly<Record<string, unknown>>;
  readonly raw?: unknown;
  readonly [key: string]: unknown;
}

export function chatResponseMetadata(partial: ChatResponseMetadata = {}): ChatResponseMetadata {
  return { ...partial };
}
