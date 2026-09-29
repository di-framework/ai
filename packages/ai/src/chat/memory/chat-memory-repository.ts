import type { Message } from '../messages/message.ts';

/**
 * Repository for storing and retrieving chat messages by conversation id.
 * Spring AI: {@code ChatMemoryRepository}.
 */
export interface ChatMemoryRepository {
  /** All known conversation ids. */
  findConversationIds(): readonly string[];

  /** Messages for a conversation (empty if none). */
  findByConversationId(conversationId: string): readonly Message[];

  /**
   * Replace all messages for the conversation with the provided list.
   */
  saveAll(conversationId: string, messages: readonly Message[]): void;

  /** Delete all messages for the conversation. */
  deleteByConversationId(conversationId: string): void;
}
