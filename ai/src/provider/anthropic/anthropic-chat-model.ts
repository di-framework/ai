import { assistantMessage, toolCall } from '../../chat/messages/factories.ts';
import type { ToolCall } from '../../chat/messages/message.ts';
import { chatResponseMetadata } from '../../chat/metadata/chat-response-metadata.ts';
import { usage } from '../../chat/metadata/usage.ts';
import type { ChatModel } from '../../chat/model/chat-model.ts';
import { ChatResponse } from '../../chat/model/chat-response.ts';
import { generation } from '../../chat/model/generation.ts';
import { mergeChatOptions } from '../../chat/prompt/chat-options.ts';
import type { Prompt } from '../../chat/prompt/prompt.ts';
import { AiError } from '../../model/errors.ts';
import {
  fetchJson,
  fetchSseJson,
  type HttpClientOptions,
  joinUrl,
  requireApiKey,
} from '../http.ts';
import type {
  AnthropicContentBlock,
  AnthropicMessagesRequest,
  AnthropicMessagesResponse,
  AnthropicStreamEvent,
} from './anthropic-api-types.ts';
import {
  type AnthropicChatOptions,
  DEFAULT_ANTHROPIC_BASE_URL,
  DEFAULT_ANTHROPIC_MAX_TOKENS,
  DEFAULT_ANTHROPIC_MESSAGES_PATH,
  DEFAULT_ANTHROPIC_MODEL,
  DEFAULT_ANTHROPIC_VERSION,
  resolveAnthropicApiKey,
} from './anthropic-chat-options.ts';
import { toAnthropicMessages, toAnthropicTools } from './map-messages.ts';

/**
 * Anthropic Messages API {@link ChatModel} (Claude family).
 *
 * Single-invocation adapter — tool loops live on ChatClient advisors.
 * Uses `fetch` only (no Anthropic SDK).
 */
export class AnthropicChatModel implements ChatModel {
  readonly options?: AnthropicChatOptions;

  constructor(options: AnthropicChatOptions = {}) {
    this.options = options;
  }

  async call(prompt: Prompt): Promise<ChatResponse> {
    const opts = this.mergeOptions(prompt);
    const body = this.buildRequest(prompt, opts, false);
    const client = this.httpClient(opts);
    const raw = (await fetchJson(client, {
      url: this.messagesUrl(opts),
      body,
      signal: opts.signal,
      headers: opts.headers,
    })) as AnthropicMessagesResponse;
    return this.toChatResponse(raw);
  }

  async *stream(prompt: Prompt): AsyncIterable<ChatResponse> {
    const opts = this.mergeOptions(prompt);
    const body = this.buildRequest(prompt, opts, true);
    const client = this.httpClient(opts);

    let id: string | undefined;
    let model: string | undefined;
    let stopReason: string | undefined;
    let textAcc = '';
    const tools = new Map<number, { id: string; name: string; arguments: string }>();
    let inputTokens = 0;
    let outputTokens = 0;

    for await (const chunk of fetchSseJson(client, {
      url: this.messagesUrl(opts),
      body,
      signal: opts.signal,
      headers: opts.headers,
    })) {
      const event = chunk as AnthropicStreamEvent;
      if (event.type === 'error') {
        throw new AiError(event.error?.message ?? 'Anthropic stream error', 'provider-error', {
          provider: 'anthropic',
          retryable: true,
        });
      }
      if (event.type === 'message_start' && event.message) {
        id = event.message.id ?? id;
        model = event.message.model ?? model;
        if (event.message.usage?.input_tokens != null) {
          inputTokens = event.message.usage.input_tokens;
        }
      }
      if (event.type === 'content_block_start' && event.content_block) {
        const index = event.index ?? 0;
        const block = event.content_block;
        if (block.type === 'tool_use') {
          tools.set(index, {
            id: block.id,
            name: block.name,
            arguments: '',
          });
        }
      }
      if (event.type === 'content_block_delta' && event.delta) {
        if (event.delta.type === 'text_delta' && event.delta.text) {
          textAcc += event.delta.text;
        }
        if (event.delta.type === 'input_json_delta' && event.delta.partial_json != null) {
          const index = event.index ?? 0;
          const existing = tools.get(index);
          if (existing) {
            existing.arguments += event.delta.partial_json;
            tools.set(index, existing);
          }
        }
      }
      if (event.type === 'message_delta') {
        if (event.delta?.stop_reason) {
          stopReason = event.delta.stop_reason;
        }
        if (event.usage?.output_tokens != null) {
          outputTokens = event.usage.output_tokens;
        }
      }

      const toolCalls = [...tools.entries()]
        .sort(([a], [b]) => a - b)
        .map(([, t]) => toolCall(t.id, t.name, t.arguments || '{}'));

      yield new ChatResponse(
        [
          generation(
            assistantMessage(textAcc || null, { toolCalls }),
            stopReason ? { finishReason: mapStopReason(stopReason) } : {},
          ),
        ],
        chatResponseMetadata({
          id,
          model,
          usage:
            inputTokens || outputTokens
              ? usage({
                  promptTokens: inputTokens,
                  completionTokens: outputTokens,
                })
              : undefined,
        }),
      );

      if (event.type === 'message_stop') return;
    }
  }

  private mergeOptions(prompt: Prompt): AnthropicChatOptions {
    return (mergeChatOptions(this.options, prompt.options) ?? {}) as AnthropicChatOptions;
  }

  private httpClient(opts: AnthropicChatOptions): HttpClientOptions {
    const apiKey = requireApiKey(
      resolveAnthropicApiKey({ ...this.options, ...opts }),
      'anthropic',
      'ANTHROPIC_API_KEY',
    );
    const version =
      opts.anthropicVersion ?? this.options?.anthropicVersion ?? DEFAULT_ANTHROPIC_VERSION;
    return {
      provider: 'anthropic',
      fetch: opts.fetch ?? this.options?.fetch,
      defaultHeaders: {
        'x-api-key': apiKey,
        'anthropic-version': version,
      },
    };
  }

  private messagesUrl(opts: AnthropicChatOptions): string {
    const base = opts.baseUrl ?? this.options?.baseUrl ?? DEFAULT_ANTHROPIC_BASE_URL;
    const path = opts.messagesPath ?? this.options?.messagesPath ?? DEFAULT_ANTHROPIC_MESSAGES_PATH;
    return joinUrl(base, path);
  }

  private buildRequest(
    prompt: Prompt,
    opts: AnthropicChatOptions,
    stream: boolean,
  ): AnthropicMessagesRequest {
    const model = opts.model ?? this.options?.model ?? DEFAULT_ANTHROPIC_MODEL;
    const mapped = toAnthropicMessages(prompt.messages);
    if (mapped.messages.length === 0) {
      throw new AiError('Prompt has no user/assistant messages for Anthropic', 'invalid-request', {
        provider: 'anthropic',
        retryable: false,
      });
    }

    const maxTokens =
      opts.maxTokens ??
      this.options?.maxTokens ??
      opts.defaultMaxTokens ??
      this.options?.defaultMaxTokens ??
      DEFAULT_ANTHROPIC_MAX_TOKENS;

    const tools = toAnthropicTools(opts.toolCallbacks);
    const request: AnthropicMessagesRequest = {
      model,
      max_tokens: maxTokens,
      messages: mapped.messages,
      stream,
    };
    if (mapped.system) request.system = mapped.system;
    if (opts.temperature !== undefined) request.temperature = opts.temperature;
    if (opts.topP !== undefined) request.top_p = opts.topP;
    if (opts.topK !== undefined) request.top_k = opts.topK;
    if (opts.stopSequences?.length) {
      request.stop_sequences = [...opts.stopSequences];
    }
    if (tools) {
      request.tools = tools;
      request.tool_choice = { type: 'auto' };
    }

    // Anthropic does not use OpenAI-style response_format; structured output
    // still works via format instructions + converters on ChatClient.
    if (opts.providerOptions) {
      Object.assign(request, opts.providerOptions);
    }

    return request;
  }

  private toChatResponse(raw: AnthropicMessagesResponse): ChatResponse {
    const { text, toolCalls } = extractContent(raw.content ?? []);
    const finishReason = mapStopReason(raw.stop_reason);

    return new ChatResponse(
      [
        generation(assistantMessage(text, { toolCalls }), {
          finishReason,
        }),
      ],
      chatResponseMetadata({
        id: raw.id,
        model: raw.model,
        usage: raw.usage
          ? usage({
              promptTokens: raw.usage.input_tokens,
              completionTokens: raw.usage.output_tokens,
              nativeUsage: raw.usage,
              cacheReadInputTokens: raw.usage.cache_read_input_tokens,
              cacheWriteInputTokens: raw.usage.cache_creation_input_tokens,
            })
          : undefined,
        raw,
      }),
    );
  }
}

function extractContent(blocks: AnthropicContentBlock[]): {
  text: string | null;
  toolCalls: ToolCall[];
} {
  const texts: string[] = [];
  const toolCalls: ToolCall[] = [];
  for (const block of blocks) {
    if (block.type === 'text') {
      texts.push(block.text);
    } else if (block.type === 'tool_use') {
      toolCalls.push(
        toolCall(
          block.id,
          block.name,
          typeof block.input === 'string' ? block.input : JSON.stringify(block.input ?? {}),
        ),
      );
    }
  }
  const joined = texts.join('');
  return {
    text: joined.length > 0 ? joined : toolCalls.length > 0 ? null : '',
    toolCalls,
  };
}

function mapStopReason(reason: string | null | undefined): string | undefined {
  if (!reason) return undefined;
  // Normalize Anthropic reasons toward common portable names.
  if (reason === 'end_turn') return 'stop';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'max_tokens') return 'length';
  return reason;
}

export function anthropicChatModel(options?: AnthropicChatOptions): AnthropicChatModel {
  return new AnthropicChatModel(options);
}
