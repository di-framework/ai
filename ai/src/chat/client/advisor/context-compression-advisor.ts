import {
  CHAT_MEMORY_CONVERSATION_ID,
  type ChatMemory,
  isReplaceableChatMemory,
  messagesEqual,
} from '../../memory/index.ts';
import {
  type ChatMessage,
  isAssistantMessage,
  isSystemMessage,
  isToolResponseMessage,
  isUserMessage,
} from '../../messages/message.ts';
import { Prompt } from '../../prompt/prompt.ts';
import { type ChatClientRequest, copyChatClientRequest } from '../chat-client-request.ts';
import type { ChatClientResponse } from '../chat-client-response.ts';
import type {
  CallAdvisor,
  CallAdvisorChain,
  StreamAdvisor,
  StreamAdvisorChain,
} from './advisor.ts';
import { HIGHEST_PRECEDENCE } from './ordered.ts';

export const DEFAULT_CONTEXT_COMPRESSION_ORDER = HIGHEST_PRECEDENCE + 225;

export interface TokenCounter {
  count(messages: readonly ChatMessage[]): number | Promise<number>;
}

export interface ContextCompressionRange {
  readonly start: number;
  readonly end: number;
}

export interface ContextCompressionRequest {
  readonly messages: readonly ChatMessage[];
  readonly tokenBudget: number;
  readonly originalTokens: number;
  readonly protectedIndices: readonly number[];
}

export interface ContextCompressionResult {
  readonly messages: readonly ChatMessage[];
  readonly compressedRanges: readonly ContextCompressionRange[];
}

export interface ContextCompressor {
  compress(
    request: ContextCompressionRequest,
  ): ContextCompressionResult | Promise<ContextCompressionResult>;
}

export type ContextCompressionPersistence = 'request' | 'memory';

export interface ContextCompressionDiagnostic {
  readonly originalTokens: number;
  readonly finalTokens: number;
  readonly compressedRanges: readonly ContextCompressionRange[];
  readonly persistence: ContextCompressionPersistence;
  readonly durationMs: number;
}

export interface ContextCompressionAdvisorOptions {
  readonly tokenBudget: number;
  readonly tokenCounter: TokenCounter;
  readonly compressor: ContextCompressor;
  readonly persistence?: ContextCompressionPersistence;
  readonly chatMemory?: ChatMemory;
  readonly order?: number;
  readonly onCompression?: (diagnostic: ContextCompressionDiagnostic) => void;
}

export type ContextCompressionErrorCode =
  | 'INVALID_REQUEST'
  | 'INVALID_RESULT'
  | 'PROTECTED_MESSAGE_REMOVED'
  | 'OVER_BUDGET'
  | 'MEMORY_NOT_REPLACEABLE';

export class ContextCompressionError extends Error {
  constructor(
    readonly code: ContextCompressionErrorCode,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = 'ContextCompressionError';
  }
}

export function parseContextCompressionRequest(value: unknown): ContextCompressionRequest {
  const request = value as Partial<ContextCompressionRequest> | undefined;
  if (
    !request ||
    !Array.isArray(request.messages) ||
    !positiveInteger(request.tokenBudget) ||
    !nonNegativeInteger(request.originalTokens) ||
    !Array.isArray(request.protectedIndices) ||
    request.protectedIndices.some(
      (index) => !nonNegativeInteger(index) || index >= (request.messages?.length ?? 0),
    )
  ) {
    throw new ContextCompressionError('INVALID_REQUEST', 'Invalid context compression request');
  }
  return request as ContextCompressionRequest;
}

export function parseContextCompressionResult(value: unknown): ContextCompressionResult {
  const result = value as Partial<ContextCompressionResult> | undefined;
  if (
    !result ||
    !Array.isArray(result.messages) ||
    result.messages.some((message) => !isChatMessage(message)) ||
    !Array.isArray(result.compressedRanges) ||
    result.compressedRanges.some(
      (range) =>
        !range ||
        !nonNegativeInteger(range.start) ||
        !nonNegativeInteger(range.end) ||
        range.end <= range.start,
    )
  ) {
    throw new ContextCompressionError('INVALID_RESULT', 'Invalid context compressor result');
  }
  return result as ContextCompressionResult;
}

/**
 * Runs after chat-memory loading and before retrieval advisors. The caller owns
 * tokenization and compression; this advisor never invokes a hidden model or
 * estimates tokens heuristically.
 */
export class ContextCompressionAdvisor implements CallAdvisor, StreamAdvisor {
  readonly name = 'Context Compression Advisor';
  readonly order: number;
  private readonly options: ContextCompressionAdvisorOptions;

  constructor(options: ContextCompressionAdvisorOptions) {
    if (!positiveInteger(options.tokenBudget)) {
      throw new ContextCompressionError(
        'INVALID_REQUEST',
        'tokenBudget must be a positive integer',
      );
    }
    if (!options.tokenCounter || !options.compressor) {
      throw new ContextCompressionError(
        'INVALID_REQUEST',
        'Both tokenCounter and compressor must be supplied',
      );
    }
    const persistence = options.persistence ?? 'request';
    if (
      persistence === 'memory' &&
      (!options.chatMemory || !isReplaceableChatMemory(options.chatMemory))
    ) {
      throw new ContextCompressionError(
        'MEMORY_NOT_REPLACEABLE',
        'Persistent context compression requires a ReplaceableChatMemory',
      );
    }
    this.options = { ...options, persistence };
    this.order = options.order ?? DEFAULT_CONTEXT_COMPRESSION_ORDER;
  }

  async before(request: ChatClientRequest): Promise<ChatClientRequest> {
    const started = performance.now();
    const messages = request.prompt.messages;
    const originalTokens = await this.options.tokenCounter.count(messages);
    assertTokenCount(originalTokens, 'original');
    if (originalTokens <= this.options.tokenBudget) return request;

    const protectedIndices = protectedMessageIndices(messages);
    const compressionRequest = parseContextCompressionRequest({
      messages,
      tokenBudget: this.options.tokenBudget,
      originalTokens,
      protectedIndices,
    });
    let result: ContextCompressionResult;
    try {
      result = parseContextCompressionResult(
        await this.options.compressor.compress(compressionRequest),
      );
    } catch (error) {
      if (error instanceof ContextCompressionError) throw error;
      throw new ContextCompressionError('INVALID_RESULT', 'Context compressor failed', {
        cause: error,
      });
    }
    validateCompressedRanges(result.compressedRanges, messages.length, protectedIndices);
    validateProtectedMessages(messages, result.messages, protectedIndices);
    const finalTokens = await this.options.tokenCounter.count(result.messages);
    assertTokenCount(finalTokens, 'final');
    if (finalTokens > this.options.tokenBudget) {
      throw new ContextCompressionError(
        'OVER_BUDGET',
        `Compressed context uses ${finalTokens} tokens; budget is ${this.options.tokenBudget}`,
      );
    }

    if (this.options.persistence === 'memory') {
      const conversationId = request.context.get(CHAT_MEMORY_CONVERSATION_ID);
      if (conversationId == null) {
        throw new ContextCompressionError(
          'INVALID_REQUEST',
          `Persistent compression requires context key '${CHAT_MEMORY_CONVERSATION_ID}'`,
        );
      }
      (this.options.chatMemory as import('../../memory/index.ts').ReplaceableChatMemory).replace(
        String(conversationId),
        result.messages,
      );
    }
    this.options.onCompression?.({
      originalTokens,
      finalTokens,
      compressedRanges: result.compressedRanges,
      persistence: this.options.persistence ?? 'request',
      durationMs: performance.now() - started,
    });
    return copyChatClientRequest(request, {
      prompt: new Prompt(result.messages, request.prompt.options),
      context: request.context,
    });
  }

  async adviseCall(
    request: ChatClientRequest,
    chain: CallAdvisorChain,
  ): Promise<ChatClientResponse> {
    return chain.nextCall(await this.before(request));
  }

  async *adviseStream(
    request: ChatClientRequest,
    chain: StreamAdvisorChain,
  ): AsyncIterable<ChatClientResponse> {
    yield* chain.nextStream(await this.before(request));
  }
}

function protectedMessageIndices(messages: readonly ChatMessage[]): number[] {
  const protectedIndices = new Set<number>();
  let lastUser = -1;
  for (let index = 0; index < messages.length; index++) {
    const message = messages[index];
    if (!message) continue;
    if (isSystemMessage(message)) protectedIndices.add(index);
    if (isUserMessage(message)) {
      lastUser = index;
      if (message.media.length > 0) protectedIndices.add(index);
    }
    if (isAssistantMessage(message)) {
      if (message.media.length > 0) protectedIndices.add(index);
      if (message.toolCalls.length > 0) {
        protectedIndices.add(index);
        let response = index + 1;
        while (
          response < messages.length &&
          isToolResponseMessage(messages[response] as ChatMessage)
        ) {
          protectedIndices.add(response++);
        }
      }
    }
  }
  if (lastUser >= 0) protectedIndices.add(lastUser);
  return [...protectedIndices].sort((left, right) => left - right);
}

function validateProtectedMessages(
  original: readonly ChatMessage[],
  compressed: readonly ChatMessage[],
  protectedIndices: readonly number[],
): void {
  let cursor = 0;
  for (const index of protectedIndices) {
    const protectedMessage = original[index];
    if (!protectedMessage) {
      throw new ContextCompressionError('INVALID_RESULT', 'Protected message index is invalid');
    }
    const found = compressed.findIndex(
      (message, compressedIndex) =>
        compressedIndex >= cursor && messagesEqual(message, protectedMessage),
    );
    if (found < 0) {
      throw new ContextCompressionError(
        'PROTECTED_MESSAGE_REMOVED',
        `Compressor removed or changed protected message at index ${index}`,
      );
    }
    cursor = found + 1;
  }
}

function validateCompressedRanges(
  ranges: readonly ContextCompressionRange[],
  messageCount: number,
  protectedIndices: readonly number[],
): void {
  let previousEnd = 0;
  for (const range of ranges) {
    if (
      range.start < previousEnd ||
      range.end > messageCount ||
      protectedIndices.some((index) => index >= range.start && index < range.end)
    ) {
      throw new ContextCompressionError(
        'INVALID_RESULT',
        'Compressed ranges overlap, exceed the input, or include protected messages',
      );
    }
    previousEnd = range.end;
  }
}

function isChatMessage(value: unknown): value is ChatMessage {
  if (!value || typeof value !== 'object') return false;
  const message = value as Partial<ChatMessage>;
  return (
    typeof message.messageType === 'string' &&
    ['system', 'user', 'assistant', 'tool'].includes(message.messageType) &&
    (typeof message.text === 'string' || message.text === null) &&
    typeof message.metadata === 'object' &&
    message.metadata !== null
  );
}

function positiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0;
}

function nonNegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

function assertTokenCount(value: number, phase: string): void {
  if (!nonNegativeInteger(value)) {
    throw new ContextCompressionError(
      'INVALID_RESULT',
      `TokenCounter returned an invalid ${phase} token count`,
    );
  }
}
