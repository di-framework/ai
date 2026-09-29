import {
  type ChatMessage,
  isAssistantMessage,
  isSystemMessage,
  isToolResponseMessage,
  isUserMessage,
} from '../../chat/messages/message.ts';
import type { Media } from '../../content/media.ts';
import type { ToolCallback } from '../../tool/tool-callback.ts';
import type {
  AnthropicContentBlock,
  AnthropicMessage,
  AnthropicTool,
} from './anthropic-api-types.ts';

export interface AnthropicMappedPrompt {
  system?: string;
  messages: AnthropicMessage[];
}

/**
 * Map portable messages to Anthropic Messages API shape.
 *
 * - System messages are concatenated into the top-level `system` field.
 * - Tool results become `user` messages with `tool_result` blocks
 *   (Anthropic does not have a `tool` role).
 * - Consecutive same-role messages are merged (API requirement).
 */
export function toAnthropicMessages(messages: readonly ChatMessage[]): AnthropicMappedPrompt {
  const systemParts: string[] = [];
  const raw: AnthropicMessage[] = [];

  for (const message of messages) {
    if (isSystemMessage(message)) {
      if (message.text) systemParts.push(message.text);
      continue;
    }
    if (isUserMessage(message)) {
      raw.push({ role: 'user', content: anthropicContent(message.text, message.media) });
      continue;
    }
    if (isAssistantMessage(message)) {
      const blocks: AnthropicContentBlock[] = [];
      if (message.text) {
        blocks.push({ type: 'text', text: message.text });
      }
      blocks.push(...message.media.map(mediaBlock));
      for (const tc of message.toolCalls) {
        let input: unknown = {};
        try {
          input = JSON.parse(tc.arguments || '{}');
        } catch {
          input = { _raw: tc.arguments };
        }
        blocks.push({
          type: 'tool_use',
          id: tc.id,
          name: tc.name,
          input,
        });
      }
      if (blocks.length === 0) {
        raw.push({ role: 'assistant', content: message.text ?? '' });
      } else {
        raw.push({ role: 'assistant', content: blocks });
      }
      continue;
    }
    if (isToolResponseMessage(message)) {
      const blocks: AnthropicContentBlock[] = message.responses.map((r) => ({
        type: 'tool_result' as const,
        tool_use_id: r.id,
        content: r.responseData,
      }));
      raw.push({ role: 'user', content: blocks });
    }
  }

  const merged = mergeConsecutiveRoles(raw);
  return {
    system: systemParts.length > 0 ? systemParts.join('\n\n') : undefined,
    messages: merged,
  };
}

function anthropicContent(
  text: string | null,
  media: readonly Media[],
): string | AnthropicContentBlock[] {
  if (!media.length) return text ?? '';
  return [...(text ? [{ type: 'text' as const, text }] : []), ...media.map(mediaBlock)];
}
function mediaBlock(item: Media): AnthropicContentBlock {
  if (typeof item.data === 'string' && item.data.startsWith('http'))
    return { type: 'image', source: { type: 'url', url: item.data } };
  const data =
    typeof item.data === 'string'
      ? item.data.replace(/^data:[^;]+;base64,/, '')
      : item.data instanceof URL
        ? item.data.toString()
        : btoa(String.fromCharCode(...item.data));
  return { type: 'image', source: { type: 'base64', media_type: item.mimeType, data } };
}

function mergeConsecutiveRoles(messages: AnthropicMessage[]): AnthropicMessage[] {
  if (messages.length === 0) return messages;
  const out: AnthropicMessage[] = [];
  for (const msg of messages) {
    const prev = out[out.length - 1];
    if (!prev || prev.role !== msg.role) {
      out.push(cloneMessage(msg));
      continue;
    }
    prev.content = concatContent(prev.content, msg.content);
  }
  return out;
}

function cloneMessage(msg: AnthropicMessage): AnthropicMessage {
  if (typeof msg.content === 'string') {
    return { role: msg.role, content: msg.content };
  }
  return { role: msg.role, content: [...msg.content] };
}

function concatContent(
  a: string | AnthropicContentBlock[],
  b: string | AnthropicContentBlock[],
): string | AnthropicContentBlock[] {
  const blocksA = toBlocks(a);
  const blocksB = toBlocks(b);
  return [...blocksA, ...blocksB];
}

function toBlocks(content: string | AnthropicContentBlock[]): AnthropicContentBlock[] {
  if (typeof content === 'string') {
    return content ? [{ type: 'text', text: content }] : [];
  }
  return content;
}

export function toAnthropicTools(
  callbacks: readonly ToolCallback[] | undefined,
): AnthropicTool[] | undefined {
  if (!callbacks || callbacks.length === 0) return undefined;
  return callbacks.map((cb) => {
    const def = cb.toolDefinition;
    let input_schema: Record<string, unknown> = {
      type: 'object',
      properties: {},
    };
    try {
      input_schema = JSON.parse(def.inputSchema) as Record<string, unknown>;
    } catch {
      // keep default
    }
    return {
      name: def.name,
      description: def.description,
      input_schema,
    };
  });
}
