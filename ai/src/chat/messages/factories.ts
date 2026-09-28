import type { Media } from '../../content/media.ts';
import type {
  AssistantMessage,
  SystemMessage,
  ToolCall,
  ToolResponse,
  ToolResponseMessage,
  UserMessage,
} from './message.ts';

export function systemMessage(
  text: string,
  metadata: Readonly<Record<string, unknown>> = {},
): SystemMessage {
  return {
    messageType: 'system',
    text,
    metadata,
  };
}

export function userMessage(
  text: string,
  options: {
    media?: readonly Media[];
    metadata?: Readonly<Record<string, unknown>>;
  } = {},
): UserMessage {
  return {
    messageType: 'user',
    text,
    media: options.media ?? [],
    metadata: options.metadata ?? {},
  };
}

export function assistantMessage(
  text: string | null,
  options: {
    toolCalls?: readonly ToolCall[];
    media?: readonly Media[];
    metadata?: Readonly<Record<string, unknown>>;
  } = {},
): AssistantMessage {
  return {
    messageType: 'assistant',
    text,
    toolCalls: options.toolCalls ?? [],
    media: options.media ?? [],
    metadata: options.metadata ?? {},
  };
}

export function toolResponseMessage(
  responses: readonly ToolResponse[],
  metadata: Readonly<Record<string, unknown>> = {},
): ToolResponseMessage {
  return {
    messageType: 'tool',
    text: null,
    responses,
    metadata,
  };
}

export function toolCall(
  id: string,
  name: string,
  args: string | Record<string, unknown>,
  type = 'function',
): ToolCall {
  return {
    id,
    type,
    name,
    arguments: typeof args === 'string' ? args : JSON.stringify(args),
  };
}

export function toolResponse(id: string, name: string, responseData: string): ToolResponse {
  return { id, name, responseData };
}
