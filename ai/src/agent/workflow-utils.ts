import type { ChatClient } from '../chat/client/default-chat-client.ts';
import type { ChatOptions } from '../chat/prompt/chat-options.ts';
import { schemaOutputConverter } from '../converter/schema-output-converter.ts';
import { AiError, cancelledError } from '../model/errors.ts';

export interface WorkflowCallOptions {
  readonly signal?: AbortSignal;
  readonly options?: ChatOptions;
}

/**
 * Throw if the abort signal is already aborted.
 */
export function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw cancelledError();
  }
}

/**
 * Call a {@link ChatClient} and return non-empty content, or throw.
 */
export async function callChatContent(
  chatClient: ChatClient,
  params: {
    user: string;
    system?: string;
    signal?: AbortSignal;
    options?: ChatOptions;
  },
): Promise<string> {
  throwIfAborted(params.signal);
  let spec = chatClient.prompt();
  if (params.system) {
    spec = spec.system(params.system);
  }
  spec = spec.user(params.user);
  const merged: ChatOptions = {
    ...params.options,
    signal: params.signal ?? params.options?.signal,
  };
  if (merged.signal !== undefined || params.options) {
    spec = spec.options(merged);
  }
  const content = await spec.call().content();
  return content ?? '';
}

/**
 * Call with structured JSON output via {@code entity()}.
 */
export async function callChatEntity<T>(
  chatClient: ChatClient,
  params: {
    user: string;
    system?: string;
    schema: Record<string, unknown>;
    signal?: AbortSignal;
    options?: ChatOptions;
    map?: (value: unknown) => T;
  },
): Promise<T> {
  throwIfAborted(params.signal);
  let spec = chatClient.prompt();
  if (params.system) {
    spec = spec.system(params.system);
  }
  spec = spec.user(params.user);
  const merged: ChatOptions = {
    ...params.options,
    signal: params.signal ?? params.options?.signal,
  };
  if (merged.signal !== undefined || params.options) {
    spec = spec.options(merged);
  }
  const converter = schemaOutputConverter<T>({
    schema: params.schema,
    map: params.map,
  });
  const entity = await spec.call().entity(converter);
  if (entity === undefined) {
    throw new AiError(
      'Workflow expected structured output but got empty content',
      'output-validation',
      {
        retryable: true,
      },
    );
  }
  return entity;
}

/**
 * Run async work with a concurrency limit (pool).
 */
export async function mapPool<T, R>(
  items: readonly T[],
  concurrency: number,
  worker: (item: T, index: number) => Promise<R>,
  signal?: AbortSignal,
): Promise<R[]> {
  if (items.length === 0) return [];
  const limit = Math.max(1, Math.min(concurrency, items.length));
  const results: R[] = new Array(items.length);
  let next = 0;

  async function runOne(): Promise<void> {
    while (true) {
      throwIfAborted(signal);
      const i = next++;
      if (i >= items.length) return;
      results[i] = await worker(items[i]!, i);
    }
  }

  await Promise.all(Array.from({ length: limit }, () => runOne()));
  return results;
}

/**
 * Lightweight JSON extraction when models wrap output in prose/fences.
 */
export function extractJsonObject(text: string): unknown {
  const cleaned = text.trim();
  try {
    return JSON.parse(cleaned);
  } catch {
    // try fenced
  }
  const fenceStart = cleaned.indexOf('```');
  if (fenceStart >= 0) {
    let contentStart = fenceStart + 3;
    if (cleaned.slice(contentStart, contentStart + 4).toLowerCase() === 'json') {
      contentStart += 4;
    }
    while (contentStart < cleaned.length && cleaned.charAt(contentStart).trim() === '') {
      contentStart++;
    }
    const fenceEnd = cleaned.indexOf('```', contentStart);
    if (fenceEnd > contentStart) {
      return JSON.parse(cleaned.slice(contentStart, fenceEnd).trim());
    }
  }
  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');
  if (start >= 0 && end > start) {
    return JSON.parse(cleaned.slice(start, end + 1));
  }
  throw new AiError(
    `Could not parse JSON from workflow response: ${cleaned.slice(0, 200)}`,
    'output-validation',
    { retryable: true },
  );
}
