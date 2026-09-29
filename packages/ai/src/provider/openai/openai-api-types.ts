/**
 * Minimal OpenAI Chat Completions wire types (compatible with Azure/Groq/etc.).
 * Not a full SDK — only what {@link OpenAiChatModel} needs.
 */

export type OpenAiMessageContent =
  | string
  | Array<
      | { type: 'text'; text: string }
      | { type: 'image_url'; image_url: { url: string; detail?: 'auto' | 'low' | 'high' } }
    >;
export interface OpenAiChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content?: OpenAiMessageContent | null;
  name?: string;
  tool_call_id?: string;
  tool_calls?: OpenAiToolCall[];
}

export interface OpenAiToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
  index?: number;
}

export interface OpenAiFunctionTool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface OpenAiChatCompletionRequest {
  model: string;
  messages: OpenAiChatMessage[];
  temperature?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  max_tokens?: number;
  max_completion_tokens?: number;
  stop?: string | string[];
  stream?: boolean;
  tools?: OpenAiFunctionTool[];
  tool_choice?: 'auto' | 'none' | 'required' | Record<string, unknown>;
  response_format?:
    | { type: 'json_object' }
    | {
        type: 'json_schema';
        json_schema: {
          name: string;
          schema: Record<string, unknown>;
          strict?: boolean;
        };
      };
  [key: string]: unknown;
}

export interface OpenAiChatCompletionResponse {
  id?: string;
  object?: string;
  created?: number;
  model?: string;
  choices?: OpenAiChoice[];
  usage?: OpenAiUsage;
}

export interface OpenAiChoice {
  index?: number;
  message?: {
    role?: string;
    content?: string | null;
    tool_calls?: OpenAiToolCall[];
    refusal?: string | null;
  };
  delta?: {
    role?: string;
    content?: string | null;
    tool_calls?: Array<{
      index?: number;
      id?: string;
      type?: string;
      function?: { name?: string; arguments?: string };
    }>;
  };
  finish_reason?: string | null;
}

export interface OpenAiUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}
