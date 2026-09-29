import type { Container as DIContainer } from '@di-framework/core/container';
import { useContainer } from '@di-framework/core/container';
import type {
  CallAdvisor,
  CallAdvisorChain,
  StreamAdvisor,
  StreamAdvisorChain,
} from '../chat/client/advisor/advisor.ts';
import { HIGHEST_PRECEDENCE } from '../chat/client/advisor/ordered.ts';
import type { ChatClientRequest } from '../chat/client/chat-client-request.ts';
import type { ChatClientResponse } from '../chat/client/chat-client-response.ts';
import type { ContainerLike } from './types.ts';

/**
 * Container event names for AI observation.
 * Subscribe with `@Subscriber(AiEvents.CHAT_RESPONSE)` on a `@Container()` bean.
 */
export const AiEvents = {
  CHAT_REQUEST: 'ai.chat.request',
  CHAT_RESPONSE: 'ai.chat.response',
  CHAT_ERROR: 'ai.chat.error',
} as const;

export type AiEventName = (typeof AiEvents)[keyof typeof AiEvents];

export interface AiChatRequestEvent {
  readonly type: typeof AiEvents.CHAT_REQUEST;
  readonly messageCount: number;
  readonly messageTypes: readonly string[];
  readonly model?: string;
  readonly hasTools: boolean;
  readonly toolNames: readonly string[];
  /** Present only when {@link ObservationAdvisorOptions.includePromptText} is true. */
  readonly promptText?: string;
  readonly startedAt: number;
}

export interface AiChatResponseEvent {
  readonly type: typeof AiEvents.CHAT_RESPONSE;
  readonly messageCount: number;
  readonly model?: string;
  readonly finishReason?: string;
  readonly usage?: {
    readonly promptTokens?: number;
    readonly completionTokens?: number;
    readonly totalTokens?: number;
  };
  readonly hasToolCalls: boolean;
  readonly durationMs: number;
  /** Present only when {@link ObservationAdvisorOptions.includeResponseText} is true. */
  readonly responseText?: string;
  readonly startedAt: number;
  readonly endedAt: number;
}

export interface AiChatErrorEvent {
  readonly type: typeof AiEvents.CHAT_ERROR;
  readonly messageCount: number;
  readonly model?: string;
  readonly errorName: string;
  readonly errorMessage: string;
  readonly durationMs: number;
  readonly startedAt: number;
  readonly endedAt: number;
}

export interface ObservationAdvisorOptions {
  readonly container?: ContainerLike | DIContainer;
  readonly order?: number;
  readonly name?: string;
  /** Include concatenated prompt texts (default false — redacted). */
  readonly includePromptText?: boolean;
  /** Include response content (default false — redacted). */
  readonly includeResponseText?: boolean;
  /**
   * Max characters when text inclusion is enabled.
   * Default 500.
   */
  readonly maxTextLength?: number;
}

/**
 * Advisor that emits redacted chat observation events on the DI container.
 * Order defaults to {@code HIGHEST_PRECEDENCE + 50} so it wraps memory + tools.
 *
 * Default payloads never include full prompts/responses — only counts, model,
 * finish reason, and token usage (Spring AI–style observation posture).
 */
export class ObservationAdvisor implements CallAdvisor, StreamAdvisor {
  readonly name: string;
  readonly order: number;

  private readonly container: ContainerLike;
  private readonly includePromptText: boolean;
  private readonly includeResponseText: boolean;
  private readonly maxTextLength: number;

  constructor(options: ObservationAdvisorOptions = {}) {
    this.container = (options.container ?? useContainer()) as ContainerLike;
    this.name = options.name ?? 'AI Observation Advisor';
    this.order = options.order ?? HIGHEST_PRECEDENCE + 50;
    this.includePromptText = options.includePromptText ?? false;
    this.includeResponseText = options.includeResponseText ?? false;
    this.maxTextLength = options.maxTextLength ?? 500;
  }

  async adviseCall(
    request: ChatClientRequest,
    chain: CallAdvisorChain,
  ): Promise<ChatClientResponse> {
    const startedAt = Date.now();
    this.emitRequest(request, startedAt);
    try {
      const response = await chain.nextCall(request);
      this.emitResponse(request, response, startedAt);
      return response;
    } catch (error) {
      this.emitError(request, error, startedAt);
      throw error;
    }
  }

  async *adviseStream(
    request: ChatClientRequest,
    chain: StreamAdvisorChain,
  ): AsyncIterable<ChatClientResponse> {
    const startedAt = Date.now();
    this.emitRequest(request, startedAt);
    let last: ChatClientResponse | undefined;
    try {
      for await (const response of chain.nextStream(request)) {
        last = response;
        yield response;
      }
      if (last) {
        this.emitResponse(request, last, startedAt);
      }
    } catch (error) {
      this.emitError(request, error, startedAt);
      throw error;
    }
  }

  private emit(event: string, payload: unknown): void {
    if (typeof this.container.emit === 'function') {
      this.container.emit(event, payload);
    }
  }

  private emitRequest(request: ChatClientRequest, startedAt: number): void {
    const messages = request.prompt.messages;
    const toolCallbacks = request.prompt.options?.toolCallbacks;
    const payload: AiChatRequestEvent = {
      type: AiEvents.CHAT_REQUEST,
      messageCount: messages.length,
      messageTypes: messages.map((m) => m.messageType),
      model: request.prompt.options?.model,
      hasTools: Boolean(toolCallbacks?.length),
      toolNames: toolCallbacks?.map((t) => t.toolDefinition.name) ?? [],
      startedAt,
    };
    if (this.includePromptText) {
      (payload as { promptText?: string }).promptText = truncate(
        messages.map((m) => m.text ?? '').join('\n'),
        this.maxTextLength,
      );
    }
    this.emit(AiEvents.CHAT_REQUEST, payload);
  }

  private emitResponse(
    request: ChatClientRequest,
    response: ChatClientResponse,
    startedAt: number,
  ): void {
    const endedAt = Date.now();
    const chat = response.chatResponse;
    const payload: AiChatResponseEvent = {
      type: AiEvents.CHAT_RESPONSE,
      messageCount: request.prompt.messages.length,
      model: chat?.metadata.model ?? request.prompt.options?.model,
      finishReason: chat?.getResult()?.metadata.finishReason,
      usage: chat?.metadata.usage
        ? {
            promptTokens: chat.metadata.usage.promptTokens,
            completionTokens: chat.metadata.usage.completionTokens,
            totalTokens: chat.metadata.usage.totalTokens,
          }
        : undefined,
      hasToolCalls: chat?.hasToolCalls() ?? false,
      durationMs: endedAt - startedAt,
      startedAt,
      endedAt,
    };
    if (this.includeResponseText) {
      (payload as { responseText?: string }).responseText = truncate(
        chat?.content ?? '',
        this.maxTextLength,
      );
    }
    this.emit(AiEvents.CHAT_RESPONSE, payload);
  }

  private emitError(request: ChatClientRequest, error: unknown, startedAt: number): void {
    const endedAt = Date.now();
    const err = error instanceof Error ? error : new Error(String(error));
    const payload: AiChatErrorEvent = {
      type: AiEvents.CHAT_ERROR,
      messageCount: request.prompt.messages.length,
      model: request.prompt.options?.model,
      errorName: err.name,
      errorMessage: truncate(err.message, this.maxTextLength),
      durationMs: endedAt - startedAt,
      startedAt,
      endedAt,
    };
    this.emit(AiEvents.CHAT_ERROR, payload);
  }
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max)}…`;
}

export function observationAdvisor(options?: ObservationAdvisorOptions): ObservationAdvisor {
  return new ObservationAdvisor(options);
}
