import { assistantMessage, toolCall } from '../../chat/messages/factories.ts';
import type { ChatMessage, ToolCall } from '../../chat/messages/message.ts';
import {
  isAssistantMessage,
  isSystemMessage,
  isToolResponseMessage,
  isUserMessage,
} from '../../chat/messages/message.ts';
import { chatResponseMetadata } from '../../chat/metadata/chat-response-metadata.ts';
import { usage } from '../../chat/metadata/usage.ts';
import type { ChatModel } from '../../chat/model/chat-model.ts';
import { ChatResponse } from '../../chat/model/chat-response.ts';
import { generation } from '../../chat/model/generation.ts';
import { type ChatOptions, mergeChatOptions } from '../../chat/prompt/chat-options.ts';
import type { Prompt } from '../../chat/prompt/prompt.ts';
import { AiError, cancelledError, isAiError } from '../../model/errors.ts';
import type { ToolCallback } from '../../tool/tool-callback.ts';
import {
  resolveWorkersAiBinding,
  type WorkersAiBindingSource,
  type WorkersAiRunOptions,
} from './binding.ts';

export const DEFAULT_WORKERS_AI_MODEL = '@cf/meta/llama-3.1-8b-instruct';

export interface WorkersAiChatOptions extends ChatOptions {
  /** Worker `env.AI`, an `{ binding }` descriptor, or a getter for the current env. */
  readonly binding?: WorkersAiBindingSource;
  /** Passed through to `binding.run` as the gateway option. */
  readonly gateway?: Readonly<Record<string, unknown>>;
}

interface WorkersAiMessage {
  readonly role: 'system' | 'user' | 'assistant' | 'tool';
  readonly content: string;
  readonly name?: string;
  readonly tool_calls?: readonly { name: string; arguments: unknown }[];
}

/**
 * {@link ChatModel} backed by a Cloudflare Workers AI binding.
 *
 * Pass the object from `env.AI`, or the `AiBindingInfo` injected by
 * `@di-framework/cloudflare`. A function binding is resolved on each call so
 * `setCloudflareBindings(env)` can happen in the Worker handler.
 */
export class WorkersAiChatModel implements ChatModel {
  readonly options?: WorkersAiChatOptions;

  constructor(options: WorkersAiChatOptions = {}) {
    this.options = options;
  }

  static of(
    binding: WorkersAiBindingSource,
    options: Omit<WorkersAiChatOptions, 'binding'> = {},
  ): WorkersAiChatModel {
    return new WorkersAiChatModel({ ...options, binding });
  }

  async call(prompt: Prompt): Promise<ChatResponse> {
    const opts = this.mergeOptions(prompt);
    this.throwIfCancelled(opts);
    const model = this.modelName(opts);
    const inputs = this.buildInputs(prompt, opts, false);
    const raw = await this.run(model, inputs, opts);
    this.throwIfCancelled(opts);
    return toChatResponse(raw, model);
  }

  async *stream(prompt: Prompt): AsyncIterable<ChatResponse> {
    const opts = this.mergeOptions(prompt);
    this.throwIfCancelled(opts);
    const model = this.modelName(opts);
    const inputs = this.buildInputs(prompt, opts, true);
    const raw = await this.run(model, inputs, opts);
    let content = '';
    let toolCalls: ToolCall[] = [];
    let finishReason: string | undefined;
    let tokenUsage: ReturnType<typeof readUsage>;
    for await (const event of iterateEvents(raw, opts.signal)) {
      this.throwIfCancelled(opts);
      if (event === DONE) continue;
      const delta = readDelta(event);
      if (delta.text) content = mergeText(content, delta.text);
      if (delta.toolCalls.length > 0) toolCalls = delta.toolCalls;
      if (delta.finishReason) finishReason = delta.finishReason;
      if (delta.usage) tokenUsage = delta.usage;
      yield snapshot(content, toolCalls, model, finishReason, tokenUsage, event);
    }
  }

  private mergeOptions(prompt: Prompt): WorkersAiChatOptions {
    return (mergeChatOptions(this.options, prompt.options) ?? {}) as WorkersAiChatOptions;
  }

  private modelName(opts: WorkersAiChatOptions): string {
    return opts.model ?? this.options?.model ?? DEFAULT_WORKERS_AI_MODEL;
  }

  private throwIfCancelled(opts: WorkersAiChatOptions): void {
    if (opts.signal?.aborted) {
      throw cancelledError('Request was cancelled', { provider: 'workers-ai' });
    }
  }

  private buildInputs(
    prompt: Prompt,
    opts: WorkersAiChatOptions,
    stream: boolean,
  ): Record<string, unknown> {
    const messages = toWorkersAiMessages(prompt.messages);
    if (messages.length === 0) {
      throw new AiError('Prompt has no messages', 'invalid-request', {
        provider: 'workers-ai',
        retryable: false,
      });
    }
    const tools = toWorkersAiTools(opts.toolCallbacks);
    const schema = parseSchema(opts.outputSchema);
    return {
      messages,
      ...(stream ? { stream: true } : {}),
      ...(opts.maxTokens !== undefined ? { max_tokens: opts.maxTokens } : {}),
      ...(opts.temperature !== undefined ? { temperature: opts.temperature } : {}),
      ...(opts.topP !== undefined ? { top_p: opts.topP } : {}),
      ...(opts.topK !== undefined ? { top_k: opts.topK } : {}),
      ...(opts.frequencyPenalty !== undefined ? { frequency_penalty: opts.frequencyPenalty } : {}),
      ...(opts.presencePenalty !== undefined ? { presence_penalty: opts.presencePenalty } : {}),
      ...(opts.stopSequences?.length ? { stop: [...opts.stopSequences] } : {}),
      ...(tools ? { tools } : {}),
      ...(schema ? { response_format: { type: 'json_schema', schema } } : {}),
    };
  }

  private async run(
    model: string,
    inputs: Record<string, unknown>,
    opts: WorkersAiChatOptions,
  ): Promise<unknown> {
    const binding = resolveWorkersAiBinding(opts.binding ?? this.options?.binding);
    try {
      return await binding.run(model, inputs, runOptions(opts));
    } catch (error) {
      if (isAiError(error)) throw error;
      if (opts.signal?.aborted) {
        throw cancelledError('Request was cancelled', { provider: 'workers-ai', model });
      }
      const message = error instanceof Error ? error.message : 'Workers AI request failed';
      throw new AiError(message, 'provider-error', {
        provider: 'workers-ai',
        model,
        cause: error,
      });
    }
  }
}

export function workersAiChatModel(options?: WorkersAiChatOptions): WorkersAiChatModel {
  return new WorkersAiChatModel(options);
}

const DONE = Symbol('workers-ai-done');

function runOptions(opts: WorkersAiChatOptions): WorkersAiRunOptions | undefined {
  const gateway =
    opts.gateway ??
    (isRecord(opts.providerOptions?.gateway) ? opts.providerOptions.gateway : undefined);
  const extraHeaders = stringRecord(opts.providerOptions?.extraHeaders);
  if (!gateway && !extraHeaders) return undefined;
  return {
    ...(gateway ? { gateway } : {}),
    ...(extraHeaders ? { extraHeaders } : {}),
  };
}

function toWorkersAiMessages(messages: readonly ChatMessage[]): WorkersAiMessage[] {
  const out: WorkersAiMessage[] = [];
  for (const message of messages) {
    if (isUserMessage(message) && message.media.length > 0) {
      throw new AiError(
        'Workers AI chat does not accept media parts on this binding',
        'invalid-request',
        {
          provider: 'workers-ai',
          retryable: false,
        },
      );
    }
    if (isSystemMessage(message) || isUserMessage(message)) {
      out.push({ role: message.messageType, content: message.text ?? '' });
      continue;
    }
    if (isAssistantMessage(message)) {
      const calls = message.toolCalls.map((call) => ({
        name: call.name,
        arguments: parseJson(call.arguments) ?? call.arguments,
      }));
      out.push({
        role: 'assistant',
        content: message.text ?? '',
        ...(calls.length > 0 ? { tool_calls: calls } : {}),
      });
      continue;
    }
    if (isToolResponseMessage(message)) {
      for (const response of message.responses) {
        out.push({
          role: 'tool',
          name: response.name,
          content: response.responseData,
        });
      }
    }
  }
  return out;
}

function toWorkersAiTools(callbacks: readonly ToolCallback[] | undefined) {
  if (!callbacks || callbacks.length === 0) return undefined;
  return callbacks.map((callback) => {
    const definition = callback.toolDefinition;
    return {
      type: 'function' as const,
      function: {
        name: definition.name,
        description: definition.description,
        parameters: parseJson(definition.inputSchema) ?? { type: 'object', properties: {} },
      },
    };
  });
}

function toChatResponse(raw: unknown, model: string): ChatResponse {
  const record = unwrap(raw);
  const choice = firstChoice(record);
  const message = isRecord(choice?.message) ? choice.message : undefined;
  const text = readText(record);
  const toolCalls = readToolCalls(message ?? record);
  const finishReason =
    typeof choice?.finish_reason === 'string'
      ? choice.finish_reason
      : typeof record.finish_reason === 'string'
        ? record.finish_reason
        : toolCalls.length > 0
          ? 'tool_calls'
          : 'stop';
  return snapshot(
    text,
    toolCalls,
    modelNameOf(record, model),
    finishReason,
    readUsage(record),
    raw,
  );
}

function snapshot(
  text: string,
  toolCalls: readonly ToolCall[],
  model: string,
  finishReason: string | undefined,
  tokenUsage: ReturnType<typeof readUsage>,
  raw: unknown,
): ChatResponse {
  return new ChatResponse(
    [
      generation(assistantMessage(text || (toolCalls.length > 0 ? null : ''), { toolCalls }), {
        ...(finishReason ? { finishReason } : {}),
      }),
    ],
    chatResponseMetadata({
      model,
      usage: tokenUsage,
      raw,
    }),
  );
}

function readDelta(event: unknown): {
  text: string;
  toolCalls: ToolCall[];
  finishReason?: string;
  usage?: ReturnType<typeof readUsage>;
} {
  const record = unwrap(event);
  const choice = firstChoice(record);
  const delta = isRecord(choice?.delta) ? choice.delta : undefined;
  const message = isRecord(choice?.message) ? choice.message : undefined;
  const text =
    typeof delta?.content === 'string'
      ? delta.content
      : typeof message?.content === 'string'
        ? message.content
        : typeof record.response === 'string'
          ? record.response
          : '';
  const finish =
    typeof choice?.finish_reason === 'string'
      ? choice.finish_reason
      : typeof record.finish_reason === 'string'
        ? record.finish_reason
        : undefined;
  return {
    text,
    toolCalls: readToolCalls(delta ?? message ?? record),
    ...(finish ? { finishReason: finish } : {}),
    usage: readUsage(record),
  };
}

function readText(record: Record<string, unknown>): string {
  if (typeof record.response === 'string') return record.response;
  const choice = firstChoice(record);
  const message = isRecord(choice?.message) ? choice.message : undefined;
  if (typeof message?.content === 'string') return message.content;
  if (typeof record.result === 'string') return record.result;
  return '';
}

function readToolCalls(source: Record<string, unknown> | undefined): ToolCall[] {
  if (!source) return [];
  const raw = source.tool_calls ?? source.toolCalls;
  if (!Array.isArray(raw)) return [];
  return raw.map((entry, index) => {
    const call = isRecord(entry) ? entry : {};
    const fn = isRecord(call.function) ? call.function : undefined;
    const name =
      typeof call.name === 'string' ? call.name : typeof fn?.name === 'string' ? fn.name : '';
    const args = call.arguments ?? fn?.arguments ?? {};
    const id =
      typeof call.id === 'string' && call.id.length > 0
        ? call.id
        : `call_${index}_${name || 'tool'}`;
    return toolCall(id, name, typeof args === 'string' || isRecord(args) ? args : String(args));
  });
}

function readUsage(record: Record<string, unknown> | undefined) {
  const native = isRecord(record?.usage) ? record.usage : undefined;
  if (!native) return undefined;
  const promptTokens =
    numberField(native, 'prompt_tokens') ?? numberField(native, 'promptTokens') ?? 0;
  const completionTokens =
    numberField(native, 'completion_tokens') ?? numberField(native, 'completionTokens') ?? 0;
  const total = numberField(native, 'total_tokens') ?? numberField(native, 'totalTokens');
  return usage({
    promptTokens,
    completionTokens,
    ...(total !== undefined ? { totalTokens: total } : {}),
    nativeUsage: native,
  });
}

function modelNameOf(record: Record<string, unknown>, fallback: string): string {
  return typeof record.model === 'string' ? record.model : fallback;
}

async function* iterateEvents(
  raw: unknown,
  signal?: AbortSignal,
): AsyncIterable<unknown | typeof DONE> {
  if (isReadableStream(raw)) {
    yield* sseFromStream(raw, signal);
    return;
  }
  if (isAsyncIterable(raw)) {
    for await (const item of raw) {
      if (signal?.aborted) {
        throw cancelledError('Request was cancelled', { provider: 'workers-ai' });
      }
      if (typeof item === 'string') {
        yield* parseSseBlock(item);
      } else {
        yield item;
      }
    }
    return;
  }
  yield raw;
}

async function* sseFromStream(
  stream: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncIterable<unknown | typeof DONE> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      if (signal?.aborted) {
        await reader.cancel();
        throw cancelledError('Request was cancelled', { provider: 'workers-ai' });
      }
      const next = await reader.read();
      if (next.done) break;
      buffer += chunkText(next.value, decoder);
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? '';
      for (const line of lines) yield* parseSseBlock(line);
    }
    buffer += decoder.decode();
    if (buffer.trim()) yield* parseSseBlock(buffer);
  } finally {
    reader.releaseLock();
  }
}

function* parseSseBlock(block: string): Generator<unknown | typeof DONE> {
  for (const line of block.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith(':') || trimmed.startsWith('event:')) continue;
    const data = trimmed.startsWith('data:') ? trimmed.slice(5).trim() : trimmed;
    if (data === '[DONE]') {
      yield DONE;
      continue;
    }
    try {
      yield JSON.parse(data) as unknown;
    } catch {
      yield { response: data };
    }
  }
}

function mergeText(current: string, next: string): string {
  if (next.startsWith(current)) return next;
  return current + next;
}

function unwrap(raw: unknown): Record<string, unknown> {
  if (!isRecord(raw)) return {};
  if (isRecord(raw.result)) return raw.result;
  return raw;
}

function firstChoice(record: Record<string, unknown>): Record<string, unknown> | undefined {
  const choices = record.choices;
  if (!Array.isArray(choices) || !isRecord(choices[0])) return undefined;
  return choices[0];
}

function parseSchema(schema: string | undefined): Record<string, unknown> | undefined {
  return parseJson(schema);
}

function parseJson(value: string | undefined): Record<string, unknown> | undefined {
  if (!value?.trim()) return undefined;
  try {
    const parsed = JSON.parse(value) as unknown;
    return isRecord(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
}

function numberField(record: Record<string, unknown>, key: string): number | undefined {
  const value = record[key];
  return typeof value === 'number' ? value : undefined;
}

function stringRecord(value: unknown): Record<string, string> | undefined {
  if (!isRecord(value)) return undefined;
  const out: Record<string, string> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (typeof entry === 'string') out[key] = entry;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

function chunkText(value: unknown, decoder: TextDecoder): string {
  if (typeof value === 'string') return value;
  if (value instanceof Uint8Array) return decoder.decode(value, { stream: true });
  return '';
}

function isReadableStream(value: unknown): value is ReadableStream<Uint8Array> {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as ReadableStream).getReader === 'function'
  );
}

function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return typeof value === 'object' && value !== null && Symbol.asyncIterator in value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
