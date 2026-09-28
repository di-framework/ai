/**
 * Minimal Anthropic Messages API wire types.
 */

export type AnthropicContentBlock =
  | { type: 'text'; text: string }
  | {
      type: 'image';
      source: { type: 'base64' | 'url'; media_type?: string; data?: string; url?: string };
    }
  | {
      type: 'tool_use';
      id: string;
      name: string;
      input: unknown;
    }
  | {
      type: 'tool_result';
      tool_use_id: string;
      content: string;
      is_error?: boolean;
    };

export interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: string | AnthropicContentBlock[];
}

export interface AnthropicTool {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
}

export interface AnthropicMessagesRequest {
  model: string;
  max_tokens: number;
  messages: AnthropicMessage[];
  system?: string | AnthropicContentBlock[];
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop_sequences?: string[];
  stream?: boolean;
  tools?: AnthropicTool[];
  tool_choice?: { type: 'auto' | 'any' | 'tool' | 'none'; name?: string };
  [key: string]: unknown;
}

export interface AnthropicMessagesResponse {
  id?: string;
  type?: string;
  role?: string;
  model?: string;
  content?: AnthropicContentBlock[];
  stop_reason?: string | null;
  stop_sequence?: string | null;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    cache_creation_input_tokens?: number;
    cache_read_input_tokens?: number;
  };
}

export type AnthropicStreamEvent =
  | { type: 'message_start'; message?: AnthropicMessagesResponse }
  | {
      type: 'content_block_start';
      index?: number;
      content_block?: AnthropicContentBlock;
    }
  | {
      type: 'content_block_delta';
      index?: number;
      delta?: {
        type?: string;
        text?: string;
        partial_json?: string;
      };
    }
  | { type: 'content_block_stop'; index?: number }
  | {
      type: 'message_delta';
      delta?: { stop_reason?: string | null };
      usage?: { output_tokens?: number };
    }
  | { type: 'message_stop' }
  | { type: 'ping' }
  | { type: 'error'; error?: { type?: string; message?: string } };
