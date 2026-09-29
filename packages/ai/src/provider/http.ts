import { AiError, type AiErrorCode, cancelledError } from '../model/errors.ts';

/** Minimal fetch surface so Node/Bun/workers and tests can inject a stub. */
export type FetchLike = (input: string | URL, init?: RequestInit) => Promise<Response>;

export interface HttpClientOptions {
  readonly fetch?: FetchLike;
  readonly defaultHeaders?: Readonly<Record<string, string>>;
  readonly provider: string;
}

export interface JsonRequestOptions {
  readonly url: string;
  readonly method?: string;
  readonly headers?: Readonly<Record<string, string>>;
  readonly body?: unknown;
  readonly signal?: AbortSignal;
}

/**
 * JSON POST/GET helper used by provider adapters.
 * Does not retry — callers or higher-level advisors own retry policy.
 */
export async function fetchJson(
  client: HttpClientOptions,
  request: JsonRequestOptions,
): Promise<unknown> {
  const response = await doFetch(client, request);
  const text = await response.text();
  if (!response.ok) {
    throw mapHttpError(client.provider, response.status, text);
  }
  if (!text) return undefined;
  try {
    return JSON.parse(text) as unknown;
  } catch (cause) {
    throw new AiError('Provider returned non-JSON body', 'provider-error', {
      provider: client.provider,
      status: response.status,
      cause,
      retryable: false,
    });
  }
}

/**
 * Streaming SSE POST. Yields parsed JSON payloads from `data:` lines.
 * OpenAI / Anthropic style: lines start with `data: ` and end with `[DONE]`.
 */
export async function* fetchSseJson(
  client: HttpClientOptions,
  request: JsonRequestOptions,
): AsyncGenerator<unknown, void, undefined> {
  const response = await doFetch(client, {
    ...request,
    headers: {
      Accept: 'text/event-stream',
      ...request.headers,
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw mapHttpError(client.provider, response.status, text);
  }

  if (!response.body) {
    throw new AiError('Streaming response has no body', 'provider-error', {
      provider: client.provider,
      retryable: false,
    });
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      if (request.signal?.aborted) {
        throw cancelledError('Request was cancelled', {
          provider: client.provider,
        });
      }
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        const payload = parseSseDataLine(line);
        if (payload === undefined) continue;
        if (payload === '[DONE]') return;
        try {
          yield JSON.parse(payload) as unknown;
        } catch {
          // Skip non-JSON keep-alives / comments.
        }
      }
    }
    if (buffer.trim()) {
      const payload = parseSseDataLine(buffer.trim());
      if (payload && payload !== '[DONE]') {
        try {
          yield JSON.parse(payload) as unknown;
        } catch {
          // ignore trailing garbage
        }
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // ignore
    }
  }
}

function parseSseDataLine(line: string): string | undefined {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith(':')) return undefined;
  if (trimmed.startsWith('data:')) {
    return trimmed.slice(5).trimStart();
  }
  return undefined;
}

async function doFetch(client: HttpClientOptions, request: JsonRequestOptions): Promise<Response> {
  if (request.signal?.aborted) {
    throw cancelledError('Request was cancelled', { provider: client.provider });
  }

  const fetchImpl = client.fetch ?? globalThis.fetch;
  if (typeof fetchImpl !== 'function') {
    throw new AiError('No fetch implementation available', 'provider-error', {
      provider: client.provider,
      retryable: false,
    });
  }

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...client.defaultHeaders,
    ...request.headers,
  };

  try {
    return await fetchImpl(request.url, {
      method: request.method ?? 'POST',
      headers,
      body: request.body === undefined ? undefined : JSON.stringify(request.body),
      signal: request.signal,
    });
  } catch (cause) {
    if (request.signal?.aborted || isAbortError(cause)) {
      throw cancelledError('Request was cancelled', {
        provider: client.provider,
        cause,
      });
    }
    throw new AiError(
      cause instanceof Error ? cause.message : 'Network request failed',
      'provider-error',
      { provider: client.provider, cause, retryable: true },
    );
  }
}

export function mapHttpError(provider: string, status: number, body: string): AiError {
  const message = extractErrorMessage(body) ?? `HTTP ${status}`;
  const code = statusToCode(status);
  return new AiError(message, code, {
    provider,
    status,
    retryable: isRetryableStatus(status),
  });
}

function statusToCode(status: number): AiErrorCode {
  if (status === 401) return 'authentication';
  if (status === 403) return 'authorization';
  if (status === 404) return 'model-unavailable';
  if (status === 408) return 'timeout';
  if (status === 429) return 'rate-limit';
  if (status >= 400 && status < 500) return 'invalid-request';
  return 'provider-error';
}

function isRetryableStatus(status: number): boolean {
  return status === 408 || status === 429 || status >= 500;
}

function extractErrorMessage(body: string): string | undefined {
  const trimmed = body.trim();
  if (!trimmed) return undefined;
  try {
    const json = JSON.parse(trimmed) as {
      error?: { message?: string; type?: string } | string;
      message?: string;
    };
    if (typeof json.error === 'string') return json.error;
    if (json.error && typeof json.error === 'object' && json.error.message) {
      return json.error.message;
    }
    if (typeof json.message === 'string') return json.message;
  } catch {
    // fall through
  }
  return trimmed.length > 400 ? `${trimmed.slice(0, 400)}…` : trimmed;
}

function isAbortError(cause: unknown): boolean {
  return cause instanceof Error && (cause.name === 'AbortError' || cause.name === 'TimeoutError');
}

export function joinUrl(baseUrl: string, path: string): string {
  let baseEnd = baseUrl.length;
  while (baseEnd > 0 && baseUrl.charCodeAt(baseEnd - 1) === 47) {
    baseEnd--;
  }
  const base = baseUrl.slice(0, baseEnd);
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${base}${p}`;
}

export function requireApiKey(
  apiKey: string | undefined,
  provider: string,
  envHint: string,
): string {
  if (apiKey?.trim()) return apiKey.trim();
  throw new AiError(
    `Missing API key for ${provider}. Set options.apiKey or ${envHint}.`,
    'authentication',
    { provider, retryable: false },
  );
}
