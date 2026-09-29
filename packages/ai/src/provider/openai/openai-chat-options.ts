import type { ChatOptions } from '../../chat/prompt/chat-options.ts';
import type { FetchLike } from '../http.ts';

/**
 * Options for {@link OpenAiChatModel}.
 * Extends portable {@link ChatOptions}; provider fields are not interpreted by core.
 */
export interface OpenAiChatOptions extends ChatOptions {
  /**
   * API key. Defaults to `process.env.OPENAI_API_KEY` when omitted.
   */
  readonly apiKey?: string;
  /**
   * Base URL without trailing slash.
   * Default: `https://api.openai.com/v1`
   * Point at Azure / Groq / Ollama OpenAI-compat endpoints as needed.
   */
  readonly baseUrl?: string;
  /**
   * Completions path relative to {@link baseUrl}.
   * Default: `/chat/completions`
   */
  readonly completionsPath?: string;
  /**
   * Optional organization header (`OpenAI-Organization`).
   */
  readonly organization?: string;
  /**
   * Optional project header (`OpenAI-Project`).
   */
  readonly project?: string;
  /**
   * Extra HTTP headers (merged last).
   */
  readonly headers?: Readonly<Record<string, string>>;
  /**
   * Injectible fetch for tests / custom transports.
   */
  readonly fetch?: FetchLike;
  /**
   * When true, use `max_completion_tokens` instead of `max_tokens`
   * (newer OpenAI models). Default false for broad compatibility.
   */
  readonly useMaxCompletionTokens?: boolean;
}

export const DEFAULT_OPENAI_BASE_URL = 'https://api.openai.com/v1';
export const DEFAULT_OPENAI_COMPLETIONS_PATH = '/chat/completions';
export const DEFAULT_OPENAI_MODEL = 'gpt-4o-mini';

export function resolveOpenAiApiKey(options?: OpenAiChatOptions): string | undefined {
  if (options?.apiKey) return options.apiKey;
  if (typeof process !== 'undefined' && process.env?.OPENAI_API_KEY) {
    return process.env.OPENAI_API_KEY;
  }
  return undefined;
}
