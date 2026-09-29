import type { Message } from '../messages/message.ts';

/**
 * Contract for storing and managing chat conversation memory.
 * Spring AI: {@code ChatMemory}.
 */
export interface ChatMemory {
  /**
   * Save messages for the given conversation.
   * The single-message overload is provided by {@link addMessage}.
   */
  add(conversationId: string, messages: readonly Message[]): void;

  /** Get messages for the conversation (empty array if none). */
  get(conversationId: string): readonly Message[];

  /** Clear all messages for the conversation. */
  clear(conversationId: string): void;
}

/** Nonbreaking extension for memories that support atomic conversation replacement. */
export interface ReplaceableChatMemory extends ChatMemory {
  replace(conversationId: string, messages: readonly Message[]): void;
}

export function isReplaceableChatMemory(memory: ChatMemory): memory is ReplaceableChatMemory {
  return typeof (memory as Partial<ReplaceableChatMemory>).replace === 'function';
}

/**
 * Context key for the chat memory conversation id.
 * Spring AI: {@code ChatMemory.CONVERSATION_ID}.
 */
export const CHAT_MEMORY_CONVERSATION_ID = 'chat_memory_conversation_id';

/**
 * Namespace matching Spring AI's {@code ChatMemory.CONVERSATION_ID} constant.
 */
export const ChatMemoryKeys = {
  CONVERSATION_ID: CHAT_MEMORY_CONVERSATION_ID,
} as const;

/** Convenience: add a single message (mirrors Spring's default method). */
export function addMessage(memory: ChatMemory, conversationId: string, message: Message): void {
  if (message == null) {
    throw new Error('message cannot be null');
  }
  memory.add(conversationId, [message]);
}
