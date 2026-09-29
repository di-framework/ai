import {
  isAssistantMessage,
  isSystemMessage,
  isToolResponseMessage,
  isUserMessage,
  type Message,
} from '../messages/message.ts';
import { MessageType } from '../messages/message-type.ts';
import { addMessage, type ReplaceableChatMemory } from './chat-memory.ts';
import type { ChatMemoryRepository } from './chat-memory-repository.ts';
import { InMemoryChatMemoryRepository } from './in-memory-chat-memory-repository.ts';

const DEFAULT_MAX_MESSAGES = 20;

export interface MessageWindowChatMemoryOptions {
  readonly chatMemoryRepository?: ChatMemoryRepository;
  /** Max messages retained (system messages are preferred during eviction). Default 20. */
  readonly maxMessages?: number;
}

/**
 * Sliding window chat memory with system-message preservation and turn-boundary
 * snapping when trimming.
 * Spring AI: {@code MessageWindowChatMemory}.
 */
export class MessageWindowChatMemory implements ReplaceableChatMemory {
  private readonly chatMemoryRepository: ChatMemoryRepository;
  private readonly maxMessages: number;

  constructor(options: MessageWindowChatMemoryOptions = {}) {
    this.chatMemoryRepository = options.chatMemoryRepository ?? new InMemoryChatMemoryRepository();
    this.maxMessages = options.maxMessages ?? DEFAULT_MAX_MESSAGES;
    if (this.maxMessages <= 0) {
      throw new Error('maxMessages must be greater than 0');
    }
  }

  static builder(options: MessageWindowChatMemoryOptions = {}): MessageWindowChatMemoryBuilder {
    return new MessageWindowChatMemoryBuilder(options);
  }

  static of(options: MessageWindowChatMemoryOptions = {}): MessageWindowChatMemory {
    return new MessageWindowChatMemory(options);
  }

  /** Spring-style single-message add. */
  addMessage(conversationId: string, message: Message): void {
    addMessage(this, conversationId, message);
  }

  add(conversationId: string, messages: readonly Message[]): void {
    assertConversationId(conversationId);
    if (messages == null) {
      throw new Error('messages cannot be null');
    }
    for (const m of messages) {
      if (m == null) {
        throw new Error('messages cannot contain null elements');
      }
    }

    const memoryMessages = this.chatMemoryRepository.findByConversationId(conversationId);
    const processed = processWindow(memoryMessages, messages, this.maxMessages);
    this.chatMemoryRepository.saveAll(conversationId, processed);
  }

  get(conversationId: string): readonly Message[] {
    assertConversationId(conversationId);
    return this.chatMemoryRepository.findByConversationId(conversationId);
  }

  clear(conversationId: string): void {
    assertConversationId(conversationId);
    this.chatMemoryRepository.deleteByConversationId(conversationId);
  }

  replace(conversationId: string, messages: readonly Message[]): void {
    assertConversationId(conversationId);
    if (messages.some((message) => message == null)) {
      throw new Error('messages cannot contain null elements');
    }
    const processed = processWindow([], messages, this.maxMessages);
    this.chatMemoryRepository.saveAll(conversationId, processed);
  }
}

export class MessageWindowChatMemoryBuilder {
  private options: MessageWindowChatMemoryOptions;

  constructor(options: MessageWindowChatMemoryOptions = {}) {
    this.options = { ...options };
  }

  chatMemoryRepository(repo: ChatMemoryRepository): this {
    this.options = { ...this.options, chatMemoryRepository: repo };
    return this;
  }

  maxMessages(max: number): this {
    this.options = { ...this.options, maxMessages: max };
    return this;
  }

  build(): MessageWindowChatMemory {
    return new MessageWindowChatMemory(this.options);
  }
}

/**
 * Process existing memory + new messages into a windowed list.
 * Exported for unit tests of eviction rules.
 */
export function processWindow(
  memoryMessages: readonly Message[],
  newMessages: readonly Message[],
  maxMessages: number,
): Message[] {
  const processed: Message[] = [];

  const hasNewSystemMessage = newMessages.some(
    (m) => isSystemMessage(m) && !memoryMessages.some((existing) => messagesEqual(existing, m)),
  );

  for (const message of memoryMessages) {
    if (hasNewSystemMessage && isSystemMessage(message)) {
      continue;
    }
    processed.push(message);
  }
  processed.push(...newMessages);

  if (processed.length <= maxMessages) {
    return processed;
  }

  // Indices of non-system messages; system messages are always preserved.
  const nonSystemIndices: number[] = [];
  for (let i = 0; i < processed.length; i++) {
    if (!isSystemMessage(processed[i]!)) {
      nonSystemIndices.push(i);
    }
  }

  // Raw cut: how many non-system messages to drop to fit maxMessages.
  let cutIndex = processed.length - maxMessages;

  // Snap forward to nearest USER so the kept window starts at a complete turn.
  while (
    cutIndex < nonSystemIndices.length &&
    processed[nonSystemIndices[cutIndex]!]?.messageType !== MessageType.USER
  ) {
    cutIndex++;
  }
  cutIndex = Math.min(cutIndex, nonSystemIndices.length);

  const removeIndices = new Set(nonSystemIndices.slice(0, cutIndex));
  const trimmed: Message[] = [];
  for (let i = 0; i < processed.length; i++) {
    if (!removeIndices.has(i)) {
      trimmed.push(processed[i]!);
    }
  }
  return trimmed;
}

/** Structural equality for messages (used by window process + memory advisor). */
export function messagesEqual(a: Message, b: Message): boolean {
  if (a === b) return true;
  if (a.messageType !== b.messageType) return false;
  if (a.text !== b.text) return false;
  if (!shallowRecordEqual(a.metadata, b.metadata)) return false;

  // Type-specific fields
  if (isUserMessage(a) && isUserMessage(b)) {
    return JSON.stringify(a.media) === JSON.stringify(b.media);
  }
  if (isAssistantMessage(a) && isAssistantMessage(b)) {
    return (
      JSON.stringify(a.toolCalls) === JSON.stringify(b.toolCalls) &&
      JSON.stringify(a.media) === JSON.stringify(b.media)
    );
  }
  if (isToolResponseMessage(a) && isToolResponseMessage(b)) {
    return JSON.stringify(a.responses) === JSON.stringify(b.responses);
  }
  return true;
}

function shallowRecordEqual(
  a: Readonly<Record<string, unknown>>,
  b: Readonly<Record<string, unknown>>,
): boolean {
  const aKeys = Object.keys(a);
  const bKeys = Object.keys(b);
  if (aKeys.length !== bKeys.length) return false;
  for (const key of aKeys) {
    if (a[key] !== b[key]) return false;
  }
  return true;
}

function assertConversationId(conversationId: string): void {
  if (conversationId == null || conversationId === '') {
    throw new Error('conversationId cannot be null or empty');
  }
}
