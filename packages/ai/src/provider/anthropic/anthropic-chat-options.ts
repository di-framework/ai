import type { ChatOptions } from '../../chat/prompt/chat-options.ts';
import type { FetchLike } from '../http.ts';

/**
 * Options for {@link AnthropicChatModel}.
 */
export interface AnthropicChatOptions extends ChatOptions {
  readonly apiKey?: string;
  /**
   * Default: `https://api.anthropic.com`
   */
  readonly baseUrl?: string;
  /**
   * Default: `/v1/messages`
   */
  readonly messagesPath?: string;
  /**
   * Anthropic API version header. Default: `2023-06-01`
   */
  readonly anthropicVersion?: string;
  readonly headers?: Readonly<Record<string, string>>;
  readonly fetch?: FetchLike;
  /**
   * Required by Anthropic when not set via maxTokens.
   * Default: 4096
   */
  readonly defaultMaxTokens?: number;
}

export const DEFAULT_ANTHROPIC_BASE_URL = 'https://api.anthropic.com';
export const DEFAULT_ANTHROPIC_MESSAGES_PATH = '/v1/messages';
export const DEFAULT_ANTHROPIC_VERSION = '2023-06-01';
export const DEFAULT_ANTHROPIC_MODEL = 'claude-sonnet-4-20250514';
export const DEFAULT_ANTHROPIC_MAX_TOKENS = 4096;

export function resolveAnthropicApiKey(options?: AnthropicChatOptions): string | undefined {
  if (options?.apiKey) return options.apiKey;
  if (typeof process !== 'undefined' && process.env?.ANTHROPIC_API_KEY) {
    return process.env.ANTHROPIC_API_KEY;
  }
  return undefined;
}
