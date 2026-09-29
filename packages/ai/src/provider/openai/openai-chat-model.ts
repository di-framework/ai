import { assistantMessage, toolCall } from '../../chat/messages/factories.ts';
import type { ToolCall } from '../../chat/messages/message.ts';
import { chatResponseMetadata } from '../../chat/metadata/chat-response-metadata.ts';
import { usage } from '../../chat/metadata/usage.ts';
import type { ChatModel } from '../../chat/model/chat-model.ts';
import { ChatResponse } from '../../chat/model/chat-response.ts';
import { generation } from '../../chat/model/generation.ts';
import { type ChatOptions, mergeChatOptions } from '../../chat/prompt/chat-options.ts';
import type { Prompt } from '../../chat/prompt/prompt.ts';
import { AiError } from '../../model/errors.ts';
import {
  fetchJson,
  fetchSseJson,
  type HttpClientOptions,
  joinUrl,
  requireApiKey,
} from '../http.ts';
import { parseJsonSchemaString, toOpenAiMessages, toOpenAiTools } from './map-messages.ts';
import type {
  OpenAiChatCompletionRequest,
  OpenAiChatCompletionResponse,
  OpenAiChoice,
  OpenAiToolCall,
} from './openai-api-types.ts';
import {
  DEFAULT_OPENAI_BASE_URL,
  DEFAULT_OPENAI_COMPLETIONS_PATH,
  DEFAULT_OPENAI_MODEL,
  type OpenAiChatOptions,
  resolveOpenAiApiKey,
} from './openai-chat-options.ts';

/**
 * OpenAI-compatible {@link ChatModel} (Chat Completions API).
 *
 * Works with OpenAI, Azure OpenAI (with path/baseUrl override), Groq, Ollama
 * OpenAI mode, OpenRouter, and other compatible gateways — no official SDK.
 *
 * Tool loops stay on {@code ChatClient} / {@code ToolCallingAdvisor}; this model
 * performs a single provider invocation per {@link call} / {@link stream}.
 */
export class OpenAiChatModel implements ChatModel {
  readonly options?: OpenAiChatOptions;

  constructor(options: OpenAiChatOptions = {}) {
    this.options = options;
  }

  async call(prompt: Prompt): Promise<ChatResponse> {
    const opts = this.mergeOptions(prompt);
    const body = this.buildRequest(prompt, opts, false);
    const client = this.httpClient(opts);
    const raw = (await fetchJson(client, {
      url: this.completionsUrl(opts),
      body,
      signal: opts.signal,
      headers: opts.headers,
    })) as OpenAiChatCompletionResponse;
    return this.toChatResponse(raw);
  }

  async *stream(prompt: Prompt): AsyncIterable<ChatResponse> {
    const opts = this.mergeOptions(prompt);
    const body = this.buildRequest(prompt, opts, true);
    const client = this.httpClient(opts);

    let id: string | undefined;
    let model: string | undefined;
    let finishReason: string | undefined;
    let contentAcc = '';
    const toolAcc = new Map<number, { id: string; name: string; arguments: string }>();

    for await (const chunk of fetchSseJson(client, {
      url: this.completionsUrl(opts),
      body,
      signal: opts.signal,
      headers: opts.headers,
    })) {
      const event = chunk as OpenAiChatCompletionResponse;
      id = event.id ?? id;
      model = event.model ?? model;
      const choice = event.choices?.[0];
      if (!choice) continue;

      if (choice.finish_reason) {
        finishReason = choice.finish_reason;
      }

      const delta = choice.delta;
      if (delta?.content) {
        contentAcc += delta.content;
      }
      if (delta?.tool_calls) {
        for (const partial of delta.tool_calls) {
          const index = partial.index ?? 0;
          const existing = toolAcc.get(index) ?? {
            id: '',
            name: '',
            arguments: '',
          };
          if (partial.id) existing.id = partial.id;
          if (partial.function?.name) {
            existing.name += partial.function.name;
          }
          if (partial.function?.arguments) {
            existing.arguments += partial.function.arguments;
          }
          toolAcc.set(index, existing);
        }
      }

      // Prefer message snapshots when a non-streaming-style chunk appears.
      if (choice.message) {
        yield this.choiceToResponse(choice, event);
        continue;
      }

      const toolCalls = [...toolAcc.entries()]
        .sort(([a], [b]) => a - b)
        .map(([, t]) => toolCall(t.id || `call_${t.name}`, t.name, t.arguments));

      yield new ChatResponse(
        [
          generation(
            assistantMessage(contentAcc || null, { toolCalls }),
            finishReason ? { finishReason } : {},
          ),
        ],
        chatResponseMetadata({ id, model }),
      );
    }
  }

  private mergeOptions(prompt: Prompt): OpenAiChatOptions {
    return (mergeChatOptions(this.options, prompt.options) ?? {}) as OpenAiChatOptions;
  }

  private httpClient(opts: OpenAiChatOptions): HttpClientOptions {
    const apiKey = requireApiKey(
      resolveOpenAiApiKey({ ...this.options, ...opts }),
      'openai',
      'OPENAI_API_KEY',
    );
    const headers: Record<string, string> = {
      Authorization: `Bearer ${apiKey}`,
    };
    if (opts.organization ?? this.options?.organization) {
      headers['OpenAI-Organization'] = String(opts.organization ?? this.options?.organization);
    }
    if (opts.project ?? this.options?.project) {
      headers['OpenAI-Project'] = String(opts.project ?? this.options?.project);
    }
    return {
      provider: 'openai',
      fetch: opts.fetch ?? this.options?.fetch,
      defaultHeaders: headers,
    };
  }

  private completionsUrl(opts: OpenAiChatOptions): string {
    const base = opts.baseUrl ?? this.options?.baseUrl ?? DEFAULT_OPENAI_BASE_URL;
    const path =
      opts.completionsPath ?? this.options?.completionsPath ?? DEFAULT_OPENAI_COMPLETIONS_PATH;
    return joinUrl(base, path);
  }

  private buildRequest(
    prompt: Prompt,
    opts: OpenAiChatOptions,
    stream: boolean,
  ): OpenAiChatCompletionRequest {
    const model = opts.model ?? this.options?.model ?? DEFAULT_OPENAI_MODEL;
    const messages = toOpenAiMessages(prompt.messages);
    if (messages.length === 0) {
      throw new AiError('Prompt has no messages', 'invalid-request', {
        provider: 'openai',
        retryable: false,
      });
    }

    const tools = toOpenAiTools(opts.toolCallbacks);
    const request: OpenAiChatCompletionRequest = {
      model,
      messages,
      stream,
    };

    if (opts.temperature !== undefined) request.temperature = opts.temperature;
    if (opts.topP !== undefined) request.top_p = opts.topP;
    if (opts.frequencyPenalty !== undefined) {
      request.frequency_penalty = opts.frequencyPenalty;
    }
    if (opts.presencePenalty !== undefined) {
      request.presence_penalty = opts.presencePenalty;
    }
    if (opts.maxTokens !== undefined) {
      if (opts.useMaxCompletionTokens ?? this.options?.useMaxCompletionTokens) {
        request.max_completion_tokens = opts.maxTokens;
      } else {
        request.max_tokens = opts.maxTokens;
      }
    }
    if (opts.stopSequences?.length) {
      request.stop =
        opts.stopSequences.length === 1 ? opts.stopSequences[0] : [...opts.stopSequences];
    }
    if (tools) {
      request.tools = tools;
      request.tool_choice = 'auto';
    }

    const schemaObj = parseJsonSchemaString(opts.outputSchema);
    if (schemaObj) {
      request.response_format = {
        type: 'json_schema',
        json_schema: {
          name: 'structured_output',
          schema: schemaObj,
          strict: true,
        },
      };
    }

    // Provider-specific escape hatch merges last (may override).
    if (opts.providerOptions) {
      Object.assign(request, opts.providerOptions);
    }

    return request;
  }

  private toChatResponse(raw: OpenAiChatCompletionResponse): ChatResponse {
    const choice = raw.choices?.[0];
    if (!choice) {
      throw new AiError('OpenAI response contained no choices', 'provider-error', {
        provider: 'openai',
        retryable: false,
        requestId: raw.id,
      });
    }
    return this.choiceToResponse(choice, raw);
  }

  private choiceToResponse(choice: OpenAiChoice, raw: OpenAiChatCompletionResponse): ChatResponse {
    const message = choice.message ?? {
      content: choice.delta?.content ?? '',
      tool_calls: undefined,
    };
    const toolCalls = (message.tool_calls ?? []).map(fromOpenAiToolCall);
    const text = message.content ?? (toolCalls.length > 0 ? null : '');
    const finishReason = choice.finish_reason ?? undefined;

    const meta = chatResponseMetadata({
      id: raw.id,
      model: raw.model,
      usage: raw.usage
        ? usage({
            promptTokens: raw.usage.prompt_tokens,
            completionTokens: raw.usage.completion_tokens,
            totalTokens: raw.usage.total_tokens,
            nativeUsage: raw.usage,
          })
        : undefined,
      raw,
    });

    return new ChatResponse(
      [
        generation(assistantMessage(text, { toolCalls }), {
          finishReason,
        }),
      ],
      meta,
    );
  }
}

function fromOpenAiToolCall(tc: OpenAiToolCall): ToolCall {
  return toolCall(
    tc.id,
    tc.function?.name ?? '',
    tc.function?.arguments ?? '{}',
    tc.type ?? 'function',
  );
}

/**
 * Factory matching Spring AI style constructors.
 */
export function openAiChatModel(options?: OpenAiChatOptions): OpenAiChatModel {
  return new OpenAiChatModel(options);
}

/** Re-export for convenience when only ChatOptions shape is needed. */
export type { ChatOptions };
