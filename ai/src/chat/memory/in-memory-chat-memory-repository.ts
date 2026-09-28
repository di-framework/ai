import type { Message } from '../messages/message.ts';
import type { ChatMemoryRepository } from './chat-memory-repository.ts';

/**
 * In-memory {@link ChatMemoryRepository}.
 * Spring AI: {@code InMemoryChatMemoryRepository}.
 */
export class InMemoryChatMemoryRepository implements ChatMemoryRepository {
  private readonly store = new Map<string, Message[]>();

  findConversationIds(): readonly string[] {
    return [...this.store.keys()];
  }

  findByConversationId(conversationId: string): readonly Message[] {
    assertConversationId(conversationId);
    const messages = this.store.get(conversationId);
    return messages ? [...messages] : [];
  }

  saveAll(conversationId: string, messages: readonly Message[]): void {
    assertConversationId(conversationId);
    if (messages == null) {
      throw new Error('messages cannot be null');
    }
    for (const m of messages) {
      if (m == null) {
        throw new Error('messages cannot contain null elements');
      }
    }
    this.store.set(conversationId, [...messages]);
  }

  deleteByConversationId(conversationId: string): void {
    assertConversationId(conversationId);
    this.store.delete(conversationId);
  }
}

function assertConversationId(conversationId: string): void {
  if (conversationId == null || conversationId === '') {
    throw new Error('conversationId cannot be null or empty');
  }
}
