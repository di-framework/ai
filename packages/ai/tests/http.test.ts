import { describe, expect, test } from 'bun:test';
import {
  fetchJson,
  fetchSseJson,
  isAiError,
  joinUrl,
  mapHttpError,
  requireApiKey,
} from '../src/index.ts';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('fetchJson', () => {
  test('returns undefined for an empty body', async () => {
    const result = await fetchJson(
      { provider: 'test', fetch: async () => new Response('', { status: 200 }) },
      { url: 'https://x.test' },
    );
    expect(result).toBeUndefined();
  });

  test('throws AiError when body is not valid JSON', async () => {
    const result = fetchJson(
      { provider: 'test', fetch: async () => new Response('not-json', { status: 200 }) },
      { url: 'https://x.test' },
    );
    await expect(result).rejects.toMatchObject({ code: 'provider-error' });
  });

  test('throws mapped error for non-ok status', async () => {
    const result = fetchJson(
      { provider: 'test', fetch: async () => jsonResponse({ message: 'boom' }, 500) },
      { url: 'https://x.test' },
    );
    await expect(result).rejects.toMatchObject({ code: 'provider-error' });
  });
});

describe('fetchSseJson', () => {
  test('throws mapped error for non-ok status', async () => {
    const gen = fetchSseJson(
      { provider: 'test', fetch: async () => jsonResponse({ message: 'bad request' }, 400) },
      { url: 'https://x.test' },
    );
    await expect(gen.next()).rejects.toMatchObject({ code: 'invalid-request' });
  });

  test('throws provider-error when the response has no body', async () => {
    const noBodyResponse = new Response(null, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    });
    Object.defineProperty(noBodyResponse, 'body', { value: null });
    const gen = fetchSseJson(
      { provider: 'test', fetch: async () => noBodyResponse },
      { url: 'https://x.test' },
    );
    await expect(gen.next()).rejects.toMatchObject({ code: 'provider-error' });
  });

  test('stops iterating once the abort signal is set mid-stream', async () => {
    const controller = new AbortController();
    let reads = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(streamController) {
        reads += 1;
        if (reads === 1) {
          controller.abort();
          streamController.enqueue(new TextEncoder().encode('data: {"a":1}\n\n'));
        } else {
          streamController.close();
        }
      },
    });
    const response = new Response(stream, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    });
    const gen = fetchSseJson(
      { provider: 'test', fetch: async () => response },
      { url: 'https://x.test', signal: controller.signal },
    );
    await expect(gen.next()).rejects.toMatchObject({ code: 'cancelled' });
  });

  test('parses a trailing buffered data line without a final newline', async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(streamController) {
        streamController.enqueue(new TextEncoder().encode('data: {"trailing":true}'));
        streamController.close();
      },
    });
    const response = new Response(stream, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    });
    const gen = fetchSseJson(
      { provider: 'test', fetch: async () => response },
      { url: 'https://x.test' },
    );
    const results: unknown[] = [];
    for await (const chunk of gen) results.push(chunk);
    expect(results).toEqual([{ trailing: true }]);
  });

  test('skips a trailing buffered line that is not valid JSON', async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(streamController) {
        streamController.enqueue(new TextEncoder().encode('data: not-json'));
        streamController.close();
      },
    });
    const response = new Response(stream, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    });
    const gen = fetchSseJson(
      { provider: 'test', fetch: async () => response },
      { url: 'https://x.test' },
    );
    const results: unknown[] = [];
    for await (const chunk of gen) results.push(chunk);
    expect(results).toEqual([]);
  });

  test('ignores non-data SSE field lines (e.g. event:/id:)', async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(streamController) {
        streamController.enqueue(
          new TextEncoder().encode('event: ping\nid: 1\ndata: {"ok":true}\n\n'),
        );
        streamController.close();
      },
    });
    const response = new Response(stream, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    });
    const gen = fetchSseJson(
      { provider: 'test', fetch: async () => response },
      { url: 'https://x.test' },
    );
    const results: unknown[] = [];
    for await (const chunk of gen) results.push(chunk);
    expect(results).toEqual([{ ok: true }]);
  });

  test('ignores a trailing [DONE] sentinel', async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(streamController) {
        streamController.enqueue(new TextEncoder().encode('data: [DONE]'));
        streamController.close();
      },
    });
    const response = new Response(stream, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    });
    const gen = fetchSseJson(
      { provider: 'test', fetch: async () => response },
      { url: 'https://x.test' },
    );
    const results: unknown[] = [];
    for await (const chunk of gen) results.push(chunk);
    expect(results).toEqual([]);
  });
});

describe('doFetch error mapping', () => {
  test('rejects immediately when the signal is already aborted', async () => {
    const controller = new AbortController();
    controller.abort();
    const result = fetchJson(
      { provider: 'test', fetch: async () => jsonResponse({}) },
      { url: 'https://x.test', signal: controller.signal },
    );
    await expect(result).rejects.toMatchObject({ code: 'cancelled' });
  });

  test('throws provider-error when no fetch implementation is available', async () => {
    const originalFetch = globalThis.fetch;
    // @ts-expect-error intentionally clearing the global for this assertion
    globalThis.fetch = undefined;
    try {
      const result = fetchJson({ provider: 'test' }, { url: 'https://x.test' });
      await expect(result).rejects.toMatchObject({ code: 'provider-error' });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('maps a generic network failure to a retryable provider-error', async () => {
    const result = fetchJson(
      {
        provider: 'test',
        fetch: async () => {
          throw new Error('network down');
        },
      },
      { url: 'https://x.test' },
    );
    await expect(result).rejects.toMatchObject({
      code: 'provider-error',
      details: { retryable: true },
    });
  });

  test('maps a thrown AbortError to a cancelled AiError', async () => {
    const abortError = new Error('aborted');
    abortError.name = 'AbortError';
    const result = fetchJson(
      {
        provider: 'test',
        fetch: async () => {
          throw abortError;
        },
      },
      { url: 'https://x.test' },
    );
    await expect(result).rejects.toMatchObject({ code: 'cancelled' });
  });

  test('maps a thrown TimeoutError to a cancelled AiError', async () => {
    const timeoutError = new Error('timed out');
    timeoutError.name = 'TimeoutError';
    const result = fetchJson(
      {
        provider: 'test',
        fetch: async () => {
          throw timeoutError;
        },
      },
      { url: 'https://x.test' },
    );
    await expect(result).rejects.toMatchObject({ code: 'cancelled' });
  });

  test('maps a non-Error network failure to a generic provider-error', async () => {
    const result = fetchJson(
      {
        provider: 'test',
        fetch: async () => {
          throw 'plain string failure';
        },
      },
      { url: 'https://x.test' },
    );
    await expect(result).rejects.toMatchObject({ code: 'provider-error' });
  });
});

describe('mapHttpError / statusToCode', () => {
  test('maps every known status family to the expected error code', () => {
    expect(mapHttpError('p', 401, '').code).toBe('authentication');
    expect(mapHttpError('p', 403, '').code).toBe('authorization');
    expect(mapHttpError('p', 404, '').code).toBe('model-unavailable');
    expect(mapHttpError('p', 408, '').code).toBe('timeout');
    expect(mapHttpError('p', 429, '').code).toBe('rate-limit');
    expect(mapHttpError('p', 418, '').code).toBe('invalid-request');
    expect(mapHttpError('p', 503, '').code).toBe('provider-error');
    expect(mapHttpError('p', 500, '').code).toBe('provider-error');
  });

  test('retryable is true only for 408/429/5xx', () => {
    expect(mapHttpError('p', 408, '').details.retryable).toBe(true);
    expect(mapHttpError('p', 429, '').details.retryable).toBe(true);
    expect(mapHttpError('p', 503, '').details.retryable).toBe(true);
    expect(mapHttpError('p', 400, '').details.retryable).toBe(false);
  });

  test('extracts error.message when error is an object', () => {
    const err = mapHttpError('p', 500, JSON.stringify({ error: { message: 'bad thing' } }));
    expect(err.message).toBe('bad thing');
  });

  test('extracts error string directly when error is a string', () => {
    const err = mapHttpError('p', 500, JSON.stringify({ error: 'bad string error' }));
    expect(err.message).toBe('bad string error');
  });

  test('falls back to top-level message field', () => {
    const err = mapHttpError('p', 500, JSON.stringify({ message: 'top level message' }));
    expect(err.message).toBe('top level message');
  });

  test('falls back to the raw body when JSON has no recognizable message', () => {
    const err = mapHttpError('p', 500, JSON.stringify({ unrelated: true }));
    expect(err.message).toBe(JSON.stringify({ unrelated: true }));
  });

  test('falls back to HTTP status when body is empty', () => {
    const err = mapHttpError('p', 502, '');
    expect(err.message).toBe('HTTP 502');
  });

  test('falls back to raw (truncated) body when it is not JSON', () => {
    const err = mapHttpError('p', 500, 'plain text error body');
    expect(err.message).toBe('plain text error body');
  });

  test('truncates very long non-JSON bodies', () => {
    const longBody = 'x'.repeat(500);
    const err = mapHttpError('p', 500, longBody);
    expect(err.message.length).toBeLessThan(420);
    expect(err.message.endsWith('…')).toBe(true);
  });
});

describe('requireApiKey', () => {
  test('returns the trimmed key when present', () => {
    expect(requireApiKey('  sk-abc  ', 'openai', 'OPENAI_API_KEY')).toBe('sk-abc');
  });

  test('throws authentication AiError when missing', () => {
    try {
      requireApiKey(undefined, 'openai', 'OPENAI_API_KEY');
      expect.unreachable();
    } catch (e) {
      expect(isAiError(e)).toBe(true);
      if (isAiError(e)) expect(e.code).toBe('authentication');
    }
  });

  test('throws authentication AiError when blank', () => {
    expect(() => requireApiKey('   ', 'openai', 'OPENAI_API_KEY')).toThrow();
  });
});

describe('joinUrl', () => {
  test('adds a leading slash to a bare path', () => {
    expect(joinUrl('https://example.test', 'v1/models')).toBe('https://example.test/v1/models');
  });
});
