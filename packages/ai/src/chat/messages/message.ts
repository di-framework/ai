import type { Media } from '../../content/media.ts';
import type { MessageType } from './message-type.ts';

/**
 * Base chat message contract, aligned with Spring AI {@code Message}.
 */
export interface Message {
  readonly messageType: MessageType;
  readonly text: string | null;
  readonly metadata: Readonly<Record<string, unknown>>;
}

/**
 * Tool call requested by the model (on an assistant message).
 * Mirrors Spring AI {@code AssistantMessage.ToolCall}.
 */
export interface ToolCall {
  readonly id: string;
  readonly type: string;
  readonly name: string;
  /** JSON-encoded arguments string, as providers typically return. */
  readonly arguments: string;
}

/**
 * One tool execution result. Mirrors Spring AI {@code ToolResponseMessage.ToolResponse}.
 */
export interface ToolResponse {
  readonly id: string;
  readonly name: string;
  readonly responseData: string;
}

export interface SystemMessage extends Message {
  readonly messageType: 'system';
}

export interface UserMessage extends Message {
  readonly messageType: 'user';
  readonly media: readonly Media[];
}

export interface AssistantMessage extends Message {
  readonly messageType: 'assistant';
  readonly toolCalls: readonly ToolCall[];
  readonly media: readonly Media[];
}

export interface ToolResponseMessage extends Message {
  readonly messageType: 'tool';
  readonly responses: readonly ToolResponse[];
}

export type ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolResponseMessage;

export function isSystemMessage(m: Message): m is SystemMessage {
  return m.messageType === 'system';
}

export function isUserMessage(m: Message): m is UserMessage {
  return m.messageType === 'user';
}

export function isAssistantMessage(m: Message): m is AssistantMessage {
  return m.messageType === 'assistant';
}

export function isToolResponseMessage(m: Message): m is ToolResponseMessage {
  return m.messageType === 'tool';
}

export function hasToolCalls(message: Message): boolean {
  return isAssistantMessage(message) && message.toolCalls.length > 0;
}
