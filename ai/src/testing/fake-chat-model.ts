import { assistantMessage } from '../chat/messages/factories.ts';
import type { ToolCall } from '../chat/messages/message.ts';
import { chatResponseMetadata } from '../chat/metadata/chat-response-metadata.ts';
import type { ChatModel } from '../chat/model/chat-model.ts';
import { ChatResponse } from '../chat/model/chat-response.ts';
import { generation } from '../chat/model/generation.ts';
import type { ChatOptions } from '../chat/prompt/chat-options.ts';
import type { Prompt } from '../chat/prompt/prompt.ts';
import { AiError, cancelledError } from '../model/errors.ts';

export type FakeChatModelHandler = (prompt: Prompt) => ChatResponse | Promise<ChatResponse>;

export interface ScriptedTurn {
  /**
   * Optional predicate; when omitted the turn always matches.
   */
  when?: (prompt: Prompt) => boolean;
  /**
   * Response for this turn. Prefer a full {@link ChatResponse} or a simple text string.
   */
  respond:
    | ChatResponse
    | string
    | ((prompt: Prompt) => ChatResponse | string | Promise<ChatResponse | string>);
}

function throwIfCancelled(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw cancelledError();
  }
}

function toResponse(value: ChatResponse | string): ChatResponse {
  return typeof value === 'string' ? ChatResponse.of(value) : value;
}

/**
 * Deterministic {@link ChatModel} for unit tests.
 * Mirrors the role of Spring AI test doubles / scripted models.
 */
export class FakeChatModel implements ChatModel {
  readonly options?: ChatOptions;
  private readonly handler: FakeChatModelHandler;
  readonly calls: Prompt[] = [];

  constructor(handler: FakeChatModelHandler | string = 'ok', options?: ChatOptions) {
    this.options = options;
    if (typeof handler === 'string') {
      const text = handler;
      this.handler = () => ChatResponse.of(text);
    } else {
      this.handler = handler;
    }
  }

  async call(prompt: Prompt): Promise<ChatResponse> {
    throwIfCancelled(prompt.options?.signal ?? this.options?.signal);
    this.calls.push(prompt);
    return this.handler(prompt);
  }

  async *stream(prompt: Prompt): AsyncIterable<ChatResponse> {
    const response = await this.call(prompt);
    const text = response.content;
    if (!text) {
      yield response;
      return;
    }
    // Emit one chunk per word for simple streaming tests.
    const parts = text.split(/(\s+)/).filter(Boolean);
    let acc = '';
    for (const part of parts) {
      throwIfCancelled(prompt.options?.signal ?? this.options?.signal);
      acc += part;
      yield ChatResponse.of(acc, response.metadata);
    }
  }
}

/**
 * Scripted model that walks through ordered turns (tool call then final answer, etc.).
 */
export class ScriptedChatModel implements ChatModel {
  readonly options?: ChatOptions;
  readonly calls: Prompt[] = [];
  private index = 0;

  constructor(
    private readonly turns: readonly ScriptedTurn[],
    options?: ChatOptions,
  ) {
    this.options = options;
  }

  async call(prompt: Prompt): Promise<ChatResponse> {
    throwIfCancelled(prompt.options?.signal ?? this.options?.signal);
    this.calls.push(prompt);

    while (this.index < this.turns.length) {
      const turn = this.turns[this.index]!;
      if (turn.when && !turn.when(prompt)) {
        this.index += 1;
        continue;
      }
      this.index += 1;
      const raw = typeof turn.respond === 'function' ? await turn.respond(prompt) : turn.respond;
      return toResponse(raw);
    }

    throw new AiError('ScriptedChatModel has no remaining turns', 'invalid-request', {
      model: 'scripted',
    });
  }
}

/**
 * Recording wrapper that delegates to another model and stores every prompt.
 */
export class RecordingChatModel implements ChatModel {
  readonly calls: Prompt[] = [];
  readonly options?: ChatOptions;

  constructor(private readonly delegate: ChatModel) {
    this.options = delegate.options;
  }

  async call(prompt: Prompt): Promise<ChatResponse> {
    this.calls.push(prompt);
    return this.delegate.call(prompt);
  }

  stream(prompt: Prompt): AsyncIterable<ChatResponse> {
    this.calls.push(prompt);
    if (!this.delegate.stream) {
      throw new AiError('Delegate does not support streaming', 'invalid-request');
    }
    return this.delegate.stream(prompt);
  }
}

/** Helpers for building scripted tool-call responses. */
export function textResponse(text: string): ChatResponse {
  return ChatResponse.of(text);
}

export function toolCallResponse(
  toolCalls: readonly ToolCall[],
  text: string | null = null,
): ChatResponse {
  return new ChatResponse(
    [generation(assistantMessage(text, { toolCalls }), { finishReason: 'tool_calls' })],
    chatResponseMetadata(),
  );
}

export function requestContains(fragment: string): (prompt: Prompt) => boolean {
  return (prompt) =>
    prompt.messages.some((m) => (m.text ?? '').toLowerCase().includes(fragment.toLowerCase()));
}
