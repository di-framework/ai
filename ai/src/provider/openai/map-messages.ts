import {
  type ChatMessage,
  isAssistantMessage,
  isSystemMessage,
  isToolResponseMessage,
  isUserMessage,
  type ToolCall,
} from '../../chat/messages/message.ts';
import type { Media } from '../../content/media.ts';
import type { ToolCallback } from '../../tool/tool-callback.ts';
import type { OpenAiChatMessage, OpenAiFunctionTool, OpenAiToolCall } from './openai-api-types.ts';

/**
 * Map portable {@link ChatMessage}s to OpenAI Chat Completions messages.
 * Tool response messages expand to one OpenAI `role: tool` message per response.
 */
export function toOpenAiMessages(messages: readonly ChatMessage[]): OpenAiChatMessage[] {
  const out: OpenAiChatMessage[] = [];
  for (const message of messages) {
    if (isSystemMessage(message)) {
      out.push({ role: 'system', content: message.text ?? '' });
      continue;
    }
    if (isUserMessage(message)) {
      out.push({ role: 'user', content: mapOpenAiContent(message.text, message.media) });
      continue;
    }
    if (isAssistantMessage(message)) {
      const toolCalls = message.toolCalls.map(toOpenAiToolCall);
      out.push({
        role: 'assistant',
        // OpenAI permits null content for tool-call-only assistant turns.
        content: message.media.length
          ? mapOpenAiContent(message.text, message.media)
          : message.text,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    if (isToolResponseMessage(message)) {
      for (const response of message.responses) {
        out.push({
          role: 'tool',
          tool_call_id: response.id,
          content: response.responseData,
          name: response.name,
        });
      }
    }
  }
  return out;
}

function mapOpenAiContent(
  text: string | null,
  media: readonly Media[],
):
  | string
  | Array<{ type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }> {
  if (!media.length) return text ?? '';
  const parts: Array<
    { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }
  > = [];
  if (text) parts.push({ type: 'text', text });
  for (const item of media) parts.push({ type: 'image_url', image_url: { url: mediaUrl(item) } });
  return parts;
}
function mediaUrl(item: Media): string {
  if (typeof item.data === 'string')
    return item.data.startsWith('data:') || item.data.startsWith('http')
      ? item.data
      : `data:${item.mimeType};base64,${item.data}`;
  if (item.data instanceof URL) return item.data.toString();
  let binary = '';
  for (const byte of item.data) binary += String.fromCharCode(byte);
  return `data:${item.mimeType};base64,${btoa(binary)}`;
}

export function toOpenAiToolCall(toolCall: ToolCall): OpenAiToolCall {
  return {
    id: toolCall.id,
    type: 'function',
    function: {
      name: toolCall.name,
      arguments: toolCall.arguments,
    },
  };
}

export function toOpenAiTools(
  callbacks: readonly ToolCallback[] | undefined,
): OpenAiFunctionTool[] | undefined {
  if (!callbacks || callbacks.length === 0) return undefined;
  return callbacks.map((cb) => {
    const def = cb.toolDefinition;
    let parameters: Record<string, unknown> = {
      type: 'object',
      properties: {},
    };
    try {
      parameters = JSON.parse(def.inputSchema) as Record<string, unknown>;
    } catch {
      // keep default
    }
    return {
      type: 'function' as const,
      function: {
        name: def.name,
        description: def.description,
        parameters,
      },
    };
  });
}

export function parseJsonSchemaString(
  schema: string | undefined,
): Record<string, unknown> | undefined {
  if (!schema?.trim()) return undefined;
  try {
    return JSON.parse(schema) as Record<string, unknown>;
  } catch {
    return undefined;
  }
}
