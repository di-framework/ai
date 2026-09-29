import { type AssistantMessage, hasToolCalls } from '../messages/message.ts';
import type { ChatResponseMetadata } from '../metadata/chat-response-metadata.ts';
import type { Generation } from './generation.ts';
import { generation } from './generation.ts';

/**
 * Chat completion response, aligned with Spring AI {@code ChatResponse}.
 *
 * Carries one or more {@link Generation}s plus optional metadata (usage, model, id).
 */
export class ChatResponse {
  readonly generations: readonly Generation[];
  readonly metadata: ChatResponseMetadata;

  constructor(generations: readonly Generation[], metadata: ChatResponseMetadata = {}) {
    this.generations = generations;
    this.metadata = metadata;
  }

  /** Spring AI {@code getResults()}. */
  get results(): readonly Generation[] {
    return this.generations;
  }

  /** First generation, if any. */
  getResult(): Generation | undefined {
    return this.generations[0];
  }

  /** First assistant message text, or empty string. */
  get content(): string {
    return this.getResult()?.output.text ?? '';
  }

  get result(): Generation | undefined {
    return this.getResult();
  }

  hasToolCalls(): boolean {
    return this.generations.some((g) => hasToolCalls(g.output));
  }

  hasFinishReasons(finishReasons: ReadonlySet<string> | readonly string[]): boolean {
    const set =
      finishReasons instanceof Set
        ? finishReasons
        : new Set([...finishReasons].map((s) => s.toLowerCase()));
    return this.generations.some((g) => {
      const reason = (g.metadata.finishReason ?? '').toLowerCase();
      return set.has(reason);
    });
  }

  static of(text: string, metadata?: ChatResponseMetadata): ChatResponse {
    return new ChatResponse([generation(text)], metadata);
  }

  static fromAssistant(
    message: AssistantMessage,
    metadata?: ChatResponseMetadata,
    generationMetadata?: Generation['metadata'],
  ): ChatResponse {
    return new ChatResponse([generation(message, generationMetadata)], metadata);
  }
}
