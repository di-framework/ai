import { describe, expect, test } from 'bun:test';
import type { CallAdvisorChain, StreamAdvisorChain } from '../src/chat/client/advisor/advisor.ts';
import { chatClientRequest } from '../src/chat/client/chat-client-request.ts';
import { chatClientResponse } from '../src/chat/client/chat-client-response.ts';
import { chatOptions, mergeChatOptions } from '../src/chat/prompt/chat-options.ts';
import { Prompt } from '../src/chat/prompt/prompt.ts';
import { withDocumentScore } from '../src/document/document.ts';
import {
  augmentWithFormatInstructions,
  ChatClientAttributes,
  createMcpToolServer,
  document,
  functionToolCallback,
  htmlDocumentLoader,
  isStandardSchema,
  isTextDocument,
  loadDocuments,
  media,
  parseStandardSchema,
  pdfDocumentLoader,
  RetryAdvisor,
  renderTemplate,
  retryAdvisor,
  StandardSchemaOutputConverter,
  standardSchemaOutputConverter,
  textDocumentLoader,
  toAnthropicMessages,
  toOpenAiMessages,
  userMessage,
} from '../src/index.ts';

describe('renderTemplate', () => {
  test('substitutes known keys and leaves unknown placeholders untouched', () => {
    expect(renderTemplate('Hello {name}, you are {age}', { name: 'Ada' })).toBe(
      'Hello Ada, you are {age}',
    );
  });

  test('renders null/undefined param values as empty string', () => {
    expect(renderTemplate('x={x}', { x: null })).toBe('x=');
    expect(renderTemplate('y={y}', { y: undefined })).toBe('y=');
  });

  test('trims whitespace inside placeholders', () => {
    expect(renderTemplate('{ name }', { name: 'Ada' })).toBe('Ada');
  });
});

describe('augmentWithFormatInstructions', () => {
  test('returns the request unchanged when a schema is set without native mode or format text', () => {
    const request = chatClientRequest(Prompt.of('q'));
    request.context.set(ChatClientAttributes.STRUCTURED_OUTPUT_SCHEMA, '{"type":"object"}');
    const result = augmentWithFormatInstructions(request);
    expect(result).toBe(request);
  });
});

describe('chatOptions / mergeChatOptions', () => {
  test('chatOptions() returns a shallow copy of the partial', () => {
    const partial = { temperature: 0.5 };
    const opts = chatOptions(partial);
    expect(opts).toEqual(partial);
    expect(opts).not.toBe(partial);
  });

  test('chatOptions() defaults to an empty object', () => {
    expect(chatOptions()).toEqual({});
  });

  test('mergeChatOptions merges tool context from both sides', () => {
    const merged = mergeChatOptions({ toolContext: { a: 1 } }, { toolContext: { b: 2 } });
    expect(merged?.toolContext).toEqual({ a: 1, b: 2 });
  });

  test('mergeChatOptions keeps base tool context when override has none', () => {
    const merged = mergeChatOptions({ toolContext: { a: 1 } }, { temperature: 0.2 });
    expect(merged?.toolContext).toEqual({ a: 1 });
  });

  test('mergeChatOptions uses override tool context when base has none', () => {
    const merged = mergeChatOptions({ temperature: 0.2 }, { toolContext: { b: 2 } });
    expect(merged?.toolContext).toEqual({ b: 2 });
  });

  test('mergeChatOptions returns undefined when both sides are absent', () => {
    expect(mergeChatOptions(undefined, undefined)).toBeUndefined();
    expect(mergeChatOptions(null, null)).toBeUndefined();
  });
});

describe('AI platform additions', () => {
  test('validates Standard Schema and converter output', async () => {
    const schema = {
      '~standard': {
        version: 1 as const,
        vendor: 'test',
        validate: (v: unknown) =>
          typeof v === 'object' && v !== null
            ? (v as { ok: boolean })
            : { issues: [{ message: 'object expected' }] },
      },
    };
    expect(await parseStandardSchema(schema, { ok: true })).toEqual({ ok: true });
    const converter = new StandardSchemaOutputConverter({ schema });
    expect(await converter.convertAsync('{"ok":true}')).toEqual({ ok: true });
    await expect(converter.convertAsync('null')).rejects.toThrow('object expected');
  });

  test('isStandardSchema recognizes valid/invalid shapes', () => {
    expect(isStandardSchema(null)).toBe(false);
    expect(isStandardSchema({})).toBe(false);
    expect(isStandardSchema({ '~standard': { version: 2, validate: () => true } })).toBe(false);
    expect(isStandardSchema({ '~standard': { version: 1, validate: 'not-a-fn' } })).toBe(false);
    expect(
      isStandardSchema({
        '~standard': { version: 1, vendor: 'test', validate: () => true },
      }),
    ).toBe(true);
  });

  test('standardSchemaOutputConverter() factory builds a working converter', async () => {
    const schema = {
      '~standard': {
        version: 1 as const,
        vendor: 'test',
        validate: (v: unknown) => v as { ok: boolean },
      },
    };
    const converter = standardSchemaOutputConverter({ schema });
    expect(converter).toBeInstanceOf(StandardSchemaOutputConverter);
    expect(await converter.convertAsync('{"ok":true}')).toEqual({ ok: true });
  });

  test('retries failed calls deterministically', async () => {
    let calls = 0;
    const chain: CallAdvisorChain = {
      callAdvisors: [],
      async nextCall() {
        calls++;
        if (calls < 3) throw { details: { retryable: true } };
        return chatClientResponse(undefined);
      },
    };
    const result = await new RetryAdvisor({ maxAttempts: 3, backoffMs: 0 }).adviseCall(
      chatClientRequest({} as never),
      chain,
    );
    expect(result.chatResponse).toBeUndefined();
    expect(calls).toBe(3);
  });

  test('retryAdvisor factory creates a working RetryAdvisor', () => {
    expect(retryAdvisor({ maxAttempts: 1 })).toBeInstanceOf(RetryAdvisor);
  });

  test('adviseCall throws immediately when shouldRetry rejects the error', async () => {
    let calls = 0;
    const chain: CallAdvisorChain = {
      callAdvisors: [],
      async nextCall() {
        calls++;
        throw new Error('permanent failure');
      },
    };
    const advisor = new RetryAdvisor({
      maxAttempts: 5,
      backoffMs: 0,
      shouldRetry: () => false,
    });
    await expect(advisor.adviseCall(chatClientRequest({} as never), chain)).rejects.toThrow(
      'permanent failure',
    );
    expect(calls).toBe(1);
  });

  test('adviseCall throws once maxAttempts is exhausted', async () => {
    let calls = 0;
    const chain: CallAdvisorChain = {
      callAdvisors: [],
      async nextCall() {
        calls++;
        throw { details: { retryable: true } };
      },
    };
    const advisor = new RetryAdvisor({ maxAttempts: 2, backoffMs: 1, jitter: 0.5 });
    await expect(advisor.adviseCall(chatClientRequest({} as never), chain)).rejects.toBeDefined();
    expect(calls).toBe(2);
  });

  test('adviseStream retries a failing stream and yields from the eventual success', async () => {
    let calls = 0;
    async function* successStream() {
      yield chatClientResponse(undefined);
    }
    const chain: StreamAdvisorChain = {
      streamAdvisors: [],
      nextStream() {
        calls++;
        if (calls < 2) {
          throw { details: { retryable: true } };
        }
        return successStream();
      },
    };
    const advisor = new RetryAdvisor({ maxAttempts: 3, backoffMs: 0 });
    const results = [];
    for await (const chunk of advisor.adviseStream(chatClientRequest({} as never), chain)) {
      results.push(chunk);
    }
    expect(results).toHaveLength(1);
    expect(calls).toBe(2);
  });

  test('adviseStream throws once maxAttempts is exhausted', async () => {
    const chain: StreamAdvisorChain = {
      streamAdvisors: [],
      async *nextStream() {
        yield undefined as never;
        throw { details: { retryable: true } };
      },
    };
    const advisor = new RetryAdvisor({ maxAttempts: 1, backoffMs: 0 });
    const iterate = async () => {
      for await (const _chunk of advisor.adviseStream(chatClientRequest({} as never), chain)) {
        // drain
      }
    };
    await expect(iterate()).rejects.toBeDefined();
  });

  test('delay actually waits and honors jitter/backoff caps', async () => {
    let calls = 0;
    const chain: CallAdvisorChain = {
      callAdvisors: [],
      async nextCall() {
        calls++;
        if (calls < 2) throw { details: { retryable: true } };
        return chatClientResponse(undefined);
      },
    };
    const start = performance.now();
    await new RetryAdvisor({
      maxAttempts: 2,
      backoffMs: 5,
      maxBackoffMs: 50,
      jitter: 0.5,
    }).adviseCall(chatClientRequest({} as never), chain);
    expect(performance.now() - start).toBeGreaterThanOrEqual(0);
    expect(calls).toBe(2);
  });

  test('delay rejects immediately if the signal is already aborted', async () => {
    const controller = new AbortController();
    controller.abort();
    let calls = 0;
    const chain: CallAdvisorChain = {
      callAdvisors: [],
      async nextCall() {
        calls++;
        throw { details: { retryable: true } };
      },
    };
    const advisor = new RetryAdvisor({
      maxAttempts: 3,
      backoffMs: 10,
      signal: controller.signal,
    });
    await expect(advisor.adviseCall(chatClientRequest({} as never), chain)).rejects.toBeDefined();
    expect(calls).toBe(1);
  });

  test('delay rejects when aborted mid-wait', async () => {
    const controller = new AbortController();
    let calls = 0;
    const chain: CallAdvisorChain = {
      callAdvisors: [],
      async nextCall() {
        calls++;
        throw { details: { retryable: true } };
      },
    };
    const advisor = new RetryAdvisor({
      maxAttempts: 3,
      backoffMs: 1000,
      signal: controller.signal,
    });
    const promise = advisor.adviseCall(chatClientRequest({} as never), chain);
    // Use a real timer (not queueMicrotask) so the abort fires strictly after
    // `delay()`'s promise executor has already run and registered its 'abort'
    // listener (several microtask hops deep from here), exercising the
    // listener callback itself rather than the synchronous pre-check.
    setTimeout(() => controller.abort(new Error('cancel wait')), 0);
    await expect(promise).rejects.toBeDefined();
    expect(calls).toBe(1);
  });

  test('maps multimodal content for both providers', () => {
    const msg = userMessage('describe', { media: [media('image/png', 'aGVsbG8=')] });
    const openai = toOpenAiMessages([msg]);
    const openAiMessage = openai[0];
    expect(openAiMessage).toBeDefined();
    expect(openAiMessage?.content).toEqual([
      { type: 'text', text: 'describe' },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,aGVsbG8=' } },
    ]);
    const anthropic = toAnthropicMessages([msg]);
    const anthropicMessage = anthropic.messages[0];
    expect(anthropicMessage).toBeDefined();
    expect(anthropicMessage?.content).toHaveLength(2);
  });

  test('serves local tools through MCP surface', async () => {
    const server = createMcpToolServer({
      tools: [
        functionToolCallback({
          name: 'add',
          inputSchema: { type: 'object' },
          call: (input: { a: number }) => input.a + 1,
        }),
      ],
    });
    expect(server.listTools()[0]?.name).toBe('add');
    expect(await server.callTool('add', { a: 2 })).toMatchObject({ content: [{ text: '3' }] });
    expect((await server.callTool('missing')).isError).toBe(true);

    expect(() =>
      server.addTool(
        functionToolCallback({
          name: 'add',
          inputSchema: { type: 'object' },
          call: (input: { a: number }) => input.a,
        }),
      ),
    ).toThrow('Duplicate MCP tool: add');

    expect(server.removeTool('add')).toBe(true);
    expect(server.removeTool('add')).toBe(false);
    expect(server.listTools()).toHaveLength(0);
  });

  test('MCP tool handler converts a thrown error into an isError content result', async () => {
    const server = createMcpToolServer({
      tools: [
        functionToolCallback({
          name: 'boom',
          inputSchema: { type: 'object' },
          call: () => {
            throw new Error('kaboom');
          },
        }),
      ],
    });
    const result = await server.callTool('boom', {});
    expect(result.isError).toBe(true);
    expect(result.content[0]).toMatchObject({
      type: 'text',
      text: expect.stringContaining('kaboom'),
    });
  });

  test('loads text, HTML, and injected PDF text', async () => {
    expect((await textDocumentLoader().load(new TextEncoder().encode('hello')))[0]?.text).toBe(
      'hello',
    );
    expect((await htmlDocumentLoader().load('<h1>Hello</h1><script>x</script>'))[0]?.text).toBe(
      'Hello',
    );
    expect((await pdfDocumentLoader(() => 'pdf text').load(new Uint8Array([1])))[0]?.text).toBe(
      'pdf text',
    );
    expect(document({ text: 'x' }).metadata).toEqual({});
  });

  test('strips HTML markup that defeats naive tag regexes', async () => {
    const load = async (html: string) => (await htmlDocumentLoader().load(html))[0]?.text;
    expect(await load('<script>evil()</script >keep')).toBe('keep');
    expect(await load('<SCRIPT\ntype="text/javascript">evil()</SCRIPT>keep')).toBe('keep');
    expect(await load('<style>a{}</style\t>keep')).toBe('keep');
    expect(await load('<div title="a > b">keep</div>')).toBe('keep');
    expect(await load('<!-- <script>evil()</script> -->keep')).toBe('keep');
    expect(await load('<script>a</script><script>b</script>keep')).toBe('keep');
    expect(await load('1 < 2 and 3 > 2')).toBe('1 < 2 and 3 > 2');
    expect(await load('<p>a&nbsp;b</p>')).toBe('a b');
    expect(await load('<title>Doc</title><p>body</p>')).toBe('Doc body');
    expect(await load('<script>unterminated')).toBe('');
    expect(await load('<!DOCTYPE html><p>keep</p>')).toBe('keep');
    expect(await load('<div class=foo>keep</div>')).toBe('keep');
    expect(await load('<div')).toBe('');
    // A raw-text element whose only closing-like tag never matches its name,
    // and which runs out of input right after the dangling "</".
    expect(await load('<script>a</')).toBe('');
  });

  test('loadDocuments loads and flattens across multiple inputs', async () => {
    const docs = await loadDocuments(textDocumentLoader(), ['a', 'b', 'c'], {
      metadata: { source: 'batch' },
    });
    expect(docs).toHaveLength(3);
    expect(docs.map((d) => d.text)).toEqual(['a', 'b', 'c']);
    expect(docs[0]?.metadata).toEqual({ source: 'batch' });
  });

  test('document() throws when both text and media are provided, or neither', () => {
    expect(() => document({ text: 'hi', media: media('text/plain', 'aGk=') })).toThrow(
      /exactly one of text or media/,
    );
    expect(() => document({ text: null, media: null })).toThrow(/exactly one of text or media/);
  });

  test('isTextDocument / withDocumentScore', () => {
    const textDoc = document({ text: 'hi' });
    const mediaDoc = document({ media: media('text/plain', 'aGk=') });
    expect(isTextDocument(textDoc)).toBe(true);
    expect(isTextDocument(mediaDoc)).toBe(false);

    const scored = withDocumentScore(textDoc, 0.9);
    expect(scored.score).toBe(0.9);
    expect(scored.id).toBe(textDoc.id);
    expect(scored.text).toBe('hi');

    const rescored = withDocumentScore(textDoc, 0.5, { id: 'custom-id', text: 'overridden' });
    expect(rescored).toMatchObject({ id: 'custom-id', text: 'overridden', score: 0.5 });
  });
});
