import type { ChatOptions } from '../prompt/chat-options.ts';
import type { Prompt } from '../prompt/prompt.ts';
import type { ChatResponse } from './chat-response.ts';

/**
 * Portable chat model, aligned with Spring AI {@code ChatModel}.
 *
 * A model performs a single provider invocation. Tool loops, memory, RAG, and
 * retries belong on {@code ChatClient} / advisors. Native subscription CLI models
 * own an internal tool loop and return completed text without pending tool calls.
 */
export interface ChatModel {
  /**
   * Default options for this model (Spring AI {@code getOptions()}).
   */
  readonly options?: ChatOptions;

  /**
   * Execute a chat completion request.
   * Spring AI: {@code ChatResponse call(Prompt prompt)}.
   */
  call(prompt: Prompt): Promise<ChatResponse>;

  /**
   * Stream chat completions as an async iterable of partial {@link ChatResponse}s.
   * Spring AI uses {@code Flux<ChatResponse>}; TypeScript uses {@link AsyncIterable}.
   *
   * Optional: models that do not stream may omit this method.
   */
  stream?(prompt: Prompt): AsyncIterable<ChatResponse>;
}

/**
 * Convenience for models that only implement streaming.
 */
export interface StreamingChatModel {
  stream(prompt: Prompt): AsyncIterable<ChatResponse>;
}
