/**
 * Message roles, matching Spring AI {@code MessageType}.
 */
export type MessageType = 'user' | 'assistant' | 'system' | 'tool';

export const MessageType = {
  USER: 'user',
  ASSISTANT: 'assistant',
  SYSTEM: 'system',
  TOOL: 'tool',
} as const satisfies Record<string, MessageType>;
