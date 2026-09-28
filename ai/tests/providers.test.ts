import { describe, expect, test } from 'bun:test';
import {
  AnthropicChatModel,
  anthropicChatModel,
  assistantMessage,
  ChatClient,
  type FetchLike,
  functionToolCallback,
  hasToolCalls,
  isAiError,
  joinUrl,
  media,
  OpenAiChatModel,
  Prompt,
  parseJsonSchemaString,
  ScriptedChatModel,
  systemMessage,
  type ToolCallback,
  toAnthropicMessages,
  toAnthropicTools,
  toOpenAiMessages,
  toOpenAiTools,
  toolCall,
  toolResponse,
  toolResponseMessage,
  userMessage,
} from '../src/index.ts';
import { openAiChatModel } from '../src/provider/openai/openai-chat-model.ts';

describe('API key resolution falls back to environment variables', () => {
  test('OpenAiChatModel reads OPENAI_API_KEY when apiKey option is omitted', async () => {
    const previous = process.env.OPENAI_API_KEY;
    process.env.OPENAI_API_KEY = 'env-openai-key';
    try {
      let seenAuth = '';
      const model = new OpenAiChatModel({
        fetch: async (_url, init) => {
          seenAuth = String((init?.headers as Record<string, string>)?.Authorization ?? '');
          return jsonResponse({ choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }] });
        },
      });
      await model.call(new Prompt('hi'));
      expect(seenAuth).toBe('Bearer env-openai-key');
    } finally {
      if (previous === undefined) delete process.env.OPENAI_API_KEY;
      else process.env.OPENAI_API_KEY = previous;
    }
  });

  test('AnthropicChatModel reads ANTHROPIC_API_KEY when apiKey option is omitted', async () => {
    const previous = process.env.ANTHROPIC_API_KEY;
    process.env.ANTHROPIC_API_KEY = 'env-anthropic-key';
    try {
      let seenHeaders: Record<string, string> = {};
      const model = new AnthropicChatModel({
        fetch: async (_url, init) => {
          seenHeaders = init?.headers as Record<string, string>;
          return jsonResponse({
            content: [{ type: 'text', text: 'hi' }],
            stop_reason: 'end_turn',
          });
        },
      });
      await model.call(new Prompt('hi'));
      expect(seenHeaders['x-api-key']).toBe('env-anthropic-key');
    } finally {
      if (previous === undefined) delete process.env.ANTHROPIC_API_KEY;
      else process.env.ANTHROPIC_API_KEY = previous;
    }
  });

  test('AnthropicChatModel requires an API key when neither option nor env var is set', async () => {
    const previous = process.env.ANTHROPIC_API_KEY;
    delete process.env.ANTHROPIC_API_KEY;
    try {
      const model = new AnthropicChatModel({ fetch: async () => jsonResponse({}) });
      await expect(model.call(new Prompt('hi'))).rejects.toMatchObject({ code: 'authentication' });
    } finally {
      if (previous !== undefined) process.env.ANTHROPIC_API_KEY = previous;
    }
  });
});

describe('joinUrl', () => {
  test('removes trailing slashes without changing internal slash runs', () => {
    expect(joinUrl('https://example.test///', '/v1/chat')).toBe('https://example.test/v1/chat');

    const slashHeavyBase = `https://example.test/${'/'.repeat(10_000)}segment`;
    expect(joinUrl(slashHeavyBase, 'v1/chat')).toBe(`${slashHeavyBase}/v1/chat`);
  });
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function sseResponse(events: string[]): Response {
  const body = `${events.map((e) => (e.startsWith('data:') ? e : `data: ${e}`)).join('\n\n')}\n\n`;
  return new Response(body, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

/** Shared ChatModel contract: call returns text, tools surface as toolCalls. */
function runChatModelContract(
  name: string,
  createTextModel: (fetch: FetchLike) => {
    call: (p: Prompt) => Promise<{ content: string; hasToolCalls: () => boolean }>;
  },
  createToolModel: (fetch: FetchLike) => {
    call: (p: Prompt) => Promise<{ hasToolCalls: () => boolean; content: string }>;
  },
  textFetch: FetchLike,
  toolFetch: FetchLike,
) {
  describe(`ChatModel contract: ${name}`, () => {
    test('call returns assistant text', async () => {
      const model = createTextModel(textFetch);
      const response = await model.call(new Prompt('Hello'));
      expect(response.content.length).toBeGreaterThan(0);
      expect(response.hasToolCalls()).toBe(false);
    });

    test('call maps tool calls', async () => {
      const model = createToolModel(toolFetch);
      const response = await model.call(new Prompt('Use a tool'));
      expect(response.hasToolCalls()).toBe(true);
    });
  });
}

describe('toOpenAiMessages', () => {
  test('maps system/user/assistant/tool', () => {
    const messages = toOpenAiMessages([
      systemMessage('sys'),
      userMessage('hi'),
      assistantMessage(null, {
        toolCalls: [toolCall('c1', 'getWeather', { city: 'Yorktown' })],
      }),
      toolResponseMessage([toolResponse('c1', 'getWeather', '{"temp":68}')]),
    ]);
    expect(messages).toEqual([
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'hi' },
      {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id: 'c1',
            type: 'function',
            function: {
              name: 'getWeather',
              arguments: '{"city":"Yorktown"}',
            },
          },
        ],
      },
      {
        role: 'tool',
        tool_call_id: 'c1',
        content: '{"temp":68}',
        name: 'getWeather',
      },
    ]);
  });
});

describe('toAnthropicMessages edge cases', () => {
  test('assistant text + tool calls both map to content blocks', () => {
    const mapped = toAnthropicMessages([
      assistantMessage('here is the result', {
        toolCalls: [toolCall('t1', 'getWeather', { city: 'NYC' })],
      }),
    ]);
    const content = mapped.messages[0]?.content;
    expect(Array.isArray(content)).toBe(true);
    const blocks = content as { type: string }[];
    expect(blocks[0]?.type).toBe('text');
    expect(blocks.some((b) => b.type === 'tool_use')).toBe(true);
  });

  test('malformed tool call arguments fall back to a _raw payload', () => {
    const badMapped = toAnthropicMessages([
      assistantMessage(null, { toolCalls: [toolCall('t2', 'x', 'not-json')] }),
    ]);
    const content = badMapped.messages[0]?.content as { input?: unknown }[] | undefined;
    const block = content?.[0];
    expect(block?.input).toEqual({ _raw: 'not-json' });
  });

  test('assistant message with no text/media/tool calls maps to empty string content', () => {
    const mapped = toAnthropicMessages([assistantMessage(null)]);
    expect(mapped.messages[0]).toEqual({ role: 'assistant', content: '' });
  });

  test('media backed by a URL instance stringifies the URL into the base64 data slot', () => {
    const mapped = toAnthropicMessages([
      userMessage('look', { media: [media('image/png', new URL('https://example.test/a.png'))] }),
    ]);
    const content = mapped.messages[0]?.content as { type: string; source?: { data?: string } }[];
    expect(content.at(-1)?.source?.data).toBe('https://example.test/a.png');
  });

  test('media backed by raw bytes is base64-encoded', () => {
    const bytes = new Uint8Array([104, 105]); // "hi"
    const mapped = toAnthropicMessages([
      userMessage('look', { media: [media('image/png', bytes)] }),
    ]);
    const content = mapped.messages[0]?.content as { source?: { data?: string } }[];
    expect(content.at(-1)?.source?.data).toBe(btoa('hi'));
  });
});

describe('toAnthropicTools', () => {
  test('returns undefined for empty/undefined callback lists', () => {
    expect(toAnthropicTools(undefined)).toBeUndefined();
    expect(toAnthropicTools([])).toBeUndefined();
  });

  test('maps tool definitions and parses their JSON input schema', () => {
    const cb = functionToolCallback({
      name: 'getWeather',
      description: 'weather lookup',
      inputSchema: { type: 'object', properties: { city: { type: 'string' } } },
      call: () => ({}),
    });
    const tools = toAnthropicTools([cb]);
    expect(tools).toHaveLength(1);
    expect(tools?.[0]?.name).toBe('getWeather');
    expect(tools?.[0]?.input_schema).toMatchObject({ type: 'object' });
  });

  test('falls back to a default object schema when inputSchema is not valid JSON', () => {
    const cb = {
      toolDefinition: {
        name: 'broken',
        description: 'broken schema',
        inputSchema: 'not-json',
      },
      call: async () => 'x',
    } as unknown as ToolCallback;
    const tools = toAnthropicTools([cb]);
    expect(tools?.[0]?.input_schema).toEqual({ type: 'object', properties: {} });
  });
});

describe('toAnthropicMessages', () => {
  test('lifts system and maps tool results to user blocks', () => {
    const mapped = toAnthropicMessages([
      systemMessage('be helpful'),
      userMessage('weather?'),
      assistantMessage(null, {
        toolCalls: [toolCall('t1', 'getWeather', { city: 'Yorktown' })],
      }),
      toolResponseMessage([toolResponse('t1', 'getWeather', '{"temp":68}')]),
    ]);
    expect(mapped.system).toBe('be helpful');
    expect(mapped.messages).toHaveLength(3);
    expect(mapped.messages[0]).toEqual({ role: 'user', content: 'weather?' });
    expect(mapped.messages[1]?.role).toBe('assistant');
    expect(mapped.messages[2]?.role).toBe('user');
    const toolResult = mapped.messages[2]?.content;
    expect(Array.isArray(toolResult)).toBe(true);
    expect((toolResult as { type: string }[])[0]?.type).toBe('tool_result');
  });

  test('merges consecutive same-role messages', () => {
    const mapped = toAnthropicMessages([
      userMessage('a'),
      toolResponseMessage([toolResponse('x', 't', 'ok')]),
    ]);
    expect(mapped.messages).toHaveLength(1);
    expect(mapped.messages[0]?.role).toBe('user');
  });
});

describe('toOpenAiMessages media edge cases', () => {
  test('media backed by a URL instance maps to the URL string directly', () => {
    const mapped = toOpenAiMessages([
      userMessage('look', { media: [media('image/png', new URL('https://example.test/a.png'))] }),
    ]);
    const content = mapped[0]?.content as { image_url: { url: string } }[];
    expect(content.at(-1)?.image_url.url).toBe('https://example.test/a.png');
  });

  test('media backed by raw bytes is base64-encoded into a data URL', () => {
    const bytes = new Uint8Array([104, 105]); // "hi"
    const mapped = toOpenAiMessages([userMessage('look', { media: [media('image/png', bytes)] })]);
    const content = mapped[0]?.content as { image_url: { url: string } }[];
    expect(content.at(-1)?.image_url.url).toBe(`data:image/png;base64,${btoa('hi')}`);
  });
});

describe('toOpenAiTools', () => {
  test('falls back to a default object schema when inputSchema is not valid JSON', () => {
    const cb = {
      toolDefinition: { name: 'broken', description: 'broken schema', inputSchema: 'not-json' },
      call: async () => 'x',
    } as unknown as ToolCallback;
    const tools = toOpenAiTools([cb]);
    expect(tools?.[0]?.function.parameters).toEqual({ type: 'object', properties: {} });
  });
});

describe('parseJsonSchemaString', () => {
  test('returns undefined for blank/undefined input', () => {
    expect(parseJsonSchemaString(undefined)).toBeUndefined();
    expect(parseJsonSchemaString('   ')).toBeUndefined();
  });

  test('returns undefined for invalid JSON', () => {
    expect(parseJsonSchemaString('not-json')).toBeUndefined();
  });

  test('parses valid JSON schema strings', () => {
    expect(parseJsonSchemaString('{"type":"object"}')).toEqual({ type: 'object' });
  });
});

describe('OpenAiChatModel', () => {
  test('openAiChatModel() factory builds a working model', async () => {
    const model = openAiChatModel({
      apiKey: 'sk-test',
      fetch: async () =>
        jsonResponse({
          choices: [{ message: { content: 'via factory' }, finish_reason: 'stop' }],
        }),
    });
    expect(model).toBeInstanceOf(OpenAiChatModel);
    const response = await model.call(new Prompt('hi'));
    expect(response.content).toBe('via factory');
  });

  test('sends chat completions request and maps response', async () => {
    let seenUrl = '';
    let seenAuth = '';
    let seenBody: Record<string, unknown> = {};

    const fetchImpl: FetchLike = async (url, init) => {
      seenUrl = String(url);
      seenAuth = String((init?.headers as Record<string, string>)?.Authorization ?? '');
      seenBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse({
        id: 'chatcmpl-1',
        model: 'gpt-4o-mini',
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: 'Hello there' },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
      });
    };

    const model = new OpenAiChatModel({
      apiKey: 'sk-test',
      model: 'gpt-4o-mini',
      fetch: fetchImpl,
    });

    const response = await model.call(
      new Prompt([systemMessage('sys'), userMessage('hi')], {
        temperature: 0.2,
      }),
    );

    expect(seenUrl).toBe('https://api.openai.com/v1/chat/completions');
    expect(seenAuth).toBe('Bearer sk-test');
    expect(seenBody.model).toBe('gpt-4o-mini');
    expect(seenBody.temperature).toBe(0.2);
    expect(seenBody.stream).toBe(false);
    expect(response.content).toBe('Hello there');
    expect(response.metadata.id).toBe('chatcmpl-1');
    expect(response.metadata.usage?.totalTokens).toBe(8);
    expect(response.getResult()?.metadata.finishReason).toBe('stop');
  });

  test('maps tool calls and tools array', async () => {
    let seenBody: Record<string, unknown> = {};
    const weather = functionToolCallback({
      name: 'getWeather',
      description: 'weather',
      inputSchema: {
        type: 'object',
        properties: { city: { type: 'string' } },
        required: ['city'],
      },
      call: () => ({ temp: 1 }),
    });

    const fetchImpl: FetchLike = async (_url, init) => {
      seenBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse({
        id: 'chatcmpl-tools',
        choices: [
          {
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [
                {
                  id: 'call_1',
                  type: 'function',
                  function: {
                    name: 'getWeather',
                    arguments: '{"city":"Yorktown"}',
                  },
                },
              ],
            },
            finish_reason: 'tool_calls',
          },
        ],
      });
    };

    const model = new OpenAiChatModel({ apiKey: 'sk', fetch: fetchImpl });
    const response = await model.call(new Prompt('weather?', { toolCallbacks: [weather] }));

    expect(Array.isArray(seenBody.tools)).toBe(true);
    const output = response.getResult()?.output;
    expect(output).toBeDefined();
    expect(hasToolCalls(output!)).toBe(true);
    expect(output!.toolCalls[0]?.name).toBe('getWeather');
  });

  test('maps HTTP 401 to authentication AiError', async () => {
    const model = new OpenAiChatModel({
      apiKey: 'bad',
      fetch: async () => jsonResponse({ error: { message: 'Incorrect API key' } }, 401),
    });
    try {
      await model.call(new Prompt('hi'));
      expect.unreachable();
    } catch (e) {
      expect(isAiError(e)).toBe(true);
      if (isAiError(e)) {
        expect(e.code).toBe('authentication');
        expect(e.details.provider).toBe('openai');
        expect(e.message).toContain('Incorrect API key');
      }
    }
  });

  test('requires api key', async () => {
    const model = new OpenAiChatModel({
      apiKey: '',
      fetch: async () => jsonResponse({}),
    });
    await expect(model.call(new Prompt('hi'))).rejects.toMatchObject({
      code: 'authentication',
    });
  });

  test('streams SSE deltas', async () => {
    const fetchImpl: FetchLike = async () =>
      sseResponse([
        JSON.stringify({
          id: 's1',
          model: 'gpt-4o-mini',
          choices: [{ delta: { content: 'Hel' }, index: 0 }],
        }),
        JSON.stringify({
          id: 's1',
          choices: [{ delta: { content: 'lo' }, index: 0 }],
        }),
        JSON.stringify({
          id: 's1',
          choices: [{ delta: {}, finish_reason: 'stop', index: 0 }],
        }),
        '[DONE]',
      ]);

    const model = new OpenAiChatModel({ apiKey: 'sk', fetch: fetchImpl });
    const chunks: string[] = [];
    for await (const chunk of model.stream(new Prompt('hi'))) {
      chunks.push(chunk.content);
    }
    expect(chunks.at(-1)).toBe('Hello');
  });

  test('streams tool_calls deltas and accumulates partial function arguments', async () => {
    const fetchImpl: FetchLike = async () =>
      sseResponse([
        JSON.stringify({
          id: 's1',
          model: 'gpt-4o-mini',
          choices: [
            {
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    id: 'call_1',
                    function: { name: 'getWeather', arguments: '{"city":' },
                  },
                ],
              },
              index: 0,
            },
          ],
        }),
        JSON.stringify({
          id: 's1',
          choices: [
            {
              delta: { tool_calls: [{ index: 0, function: { arguments: '"Yorktown"}' } }] },
              index: 0,
            },
          ],
        }),
        JSON.stringify({
          id: 's1',
          choices: [{ delta: {}, finish_reason: 'tool_calls', index: 0 }],
        }),
        '[DONE]',
      ]);

    const model = new OpenAiChatModel({ apiKey: 'sk', fetch: fetchImpl });
    let last: Awaited<ReturnType<typeof model.call>> | undefined;
    for await (const chunk of model.stream(new Prompt('weather?'))) {
      last = chunk;
    }
    expect(last?.getResult()?.output.toolCalls).toEqual([
      { id: 'call_1', type: 'function', name: 'getWeather', arguments: '{"city":"Yorktown"}' },
    ]);
    expect(last?.getResult()?.metadata.finishReason).toBe('tool_calls');
  });

  test('honors AbortSignal', async () => {
    const controller = new AbortController();
    controller.abort();
    const model = new OpenAiChatModel({
      apiKey: 'sk',
      fetch: async () => jsonResponse({ choices: [] }),
    });
    await expect(model.call(new Prompt('hi', { signal: controller.signal }))).rejects.toMatchObject(
      { code: 'cancelled' },
    );
  });

  test('response_format from outputSchema', async () => {
    let seenBody: Record<string, unknown> = {};
    const model = new OpenAiChatModel({
      apiKey: 'sk',
      fetch: async (_u, init) => {
        seenBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
        return jsonResponse({
          choices: [{ message: { content: '{"a":1}' }, finish_reason: 'stop' }],
        });
      },
    });
    await model.call(
      new Prompt('x', {
        outputSchema: JSON.stringify({
          type: 'object',
          properties: { a: { type: 'number' } },
        }),
      }),
    );
    expect(seenBody.response_format).toMatchObject({ type: 'json_schema' });
  });
});

describe('AnthropicChatModel', () => {
  test('anthropicChatModel() factory builds a working model', async () => {
    const model = anthropicChatModel({
      apiKey: 'ant-key',
      fetch: async () =>
        jsonResponse({
          content: [{ type: 'text', text: 'via factory' }],
          stop_reason: 'end_turn',
        }),
    });
    expect(model).toBeInstanceOf(AnthropicChatModel);
    const response = await model.call(new Prompt('hi'));
    expect(response.content).toBe('via factory');
  });

  test('sends messages request and maps text response', async () => {
    let seenUrl = '';
    let seenHeaders: Record<string, string> = {};
    let seenBody: Record<string, unknown> = {};

    const fetchImpl: FetchLike = async (url, init) => {
      seenUrl = String(url);
      seenHeaders = init?.headers as Record<string, string>;
      seenBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse({
        id: 'msg_1',
        model: 'claude-sonnet-4-20250514',
        role: 'assistant',
        content: [{ type: 'text', text: 'Hi from Claude' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 4 },
      });
    };

    const model = new AnthropicChatModel({
      apiKey: 'ant-key',
      fetch: fetchImpl,
    });

    const response = await model.call(new Prompt([systemMessage('sys'), userMessage('hello')]));

    expect(seenUrl).toBe('https://api.anthropic.com/v1/messages');
    expect(seenHeaders['x-api-key']).toBe('ant-key');
    expect(seenHeaders['anthropic-version']).toBe('2023-06-01');
    expect(seenBody.system).toBe('sys');
    expect(seenBody.max_tokens).toBe(4096);
    expect(response.content).toBe('Hi from Claude');
    expect(response.getResult()?.metadata.finishReason).toBe('stop');
    expect(response.metadata.usage?.promptTokens).toBe(10);
  });

  test('maps tool_use blocks', async () => {
    const model = new AnthropicChatModel({
      apiKey: 'k',
      fetch: async () =>
        jsonResponse({
          content: [
            {
              type: 'tool_use',
              id: 'toolu_1',
              name: 'getWeather',
              input: { city: 'Yorktown' },
            },
          ],
          stop_reason: 'tool_use',
        }),
    });

    const response = await model.call(new Prompt('weather'));
    expect(response.hasToolCalls()).toBe(true);
    expect(response.getResult()?.output.toolCalls[0]).toMatchObject({
      id: 'toolu_1',
      name: 'getWeather',
    });
    expect(response.getResult()?.metadata.finishReason).toBe('tool_calls');
  });

  test('streams text deltas', async () => {
    const fetchImpl: FetchLike = async () =>
      sseResponse([
        JSON.stringify({
          type: 'message_start',
          message: {
            id: 'm1',
            model: 'claude',
            usage: { input_tokens: 1, output_tokens: 0 },
          },
        }),
        JSON.stringify({
          type: 'content_block_delta',
          index: 0,
          delta: { type: 'text_delta', text: 'Hey' },
        }),
        JSON.stringify({
          type: 'content_block_delta',
          index: 0,
          delta: { type: 'text_delta', text: '!' },
        }),
        JSON.stringify({
          type: 'message_delta',
          delta: { stop_reason: 'end_turn' },
          usage: { output_tokens: 2 },
        }),
        JSON.stringify({ type: 'message_stop' }),
      ]);

    const model = new AnthropicChatModel({ apiKey: 'k', fetch: fetchImpl });
    let last = '';
    for await (const chunk of model.stream(new Prompt('hi'))) {
      last = chunk.content;
    }
    expect(last).toBe('Hey!');
  });

  test('streams tool_use blocks and accumulates partial JSON arguments', async () => {
    const fetchImpl: FetchLike = async () =>
      sseResponse([
        JSON.stringify({ type: 'message_start', message: { id: 'm1', model: 'claude' } }),
        JSON.stringify({
          type: 'content_block_start',
          index: 0,
          content_block: { type: 'tool_use', id: 'toolu_1', name: 'getWeather' },
        }),
        JSON.stringify({
          type: 'content_block_delta',
          index: 0,
          delta: { type: 'input_json_delta', partial_json: '{"city":' },
        }),
        JSON.stringify({
          type: 'content_block_delta',
          index: 0,
          delta: { type: 'input_json_delta', partial_json: '"Yorktown"}' },
        }),
        JSON.stringify({
          type: 'message_delta',
          delta: { stop_reason: 'tool_use' },
          usage: { output_tokens: 3 },
        }),
        JSON.stringify({ type: 'message_stop' }),
      ]);

    const model = new AnthropicChatModel({ apiKey: 'k', fetch: fetchImpl });
    let last: Awaited<ReturnType<typeof model.call>> | undefined;
    for await (const chunk of model.stream(new Prompt('weather?'))) {
      last = chunk;
    }
    expect(last?.getResult()?.output.toolCalls).toEqual([
      { id: 'toolu_1', type: 'function', name: 'getWeather', arguments: '{"city":"Yorktown"}' },
    ]);
    expect(last?.getResult()?.metadata.finishReason).toBe('tool_calls');
  });

  test('maps 429 rate limit', async () => {
    const model = new AnthropicChatModel({
      apiKey: 'k',
      fetch: async () => jsonResponse({ error: { message: 'rate limited' } }, 429),
    });
    await expect(model.call(new Prompt('x'))).rejects.toMatchObject({
      code: 'rate-limit',
      details: { retryable: true },
    });
  });
});

// Shared contract suite for both providers with mock transports.
runChatModelContract(
  'openai',
  (fetch) => new OpenAiChatModel({ apiKey: 'k', fetch }),
  (fetch) => new OpenAiChatModel({ apiKey: 'k', fetch }),
  async () =>
    jsonResponse({
      choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }],
    }),
  async () =>
    jsonResponse({
      choices: [
        {
          message: {
            content: null,
            tool_calls: [
              {
                id: 'c',
                type: 'function',
                function: { name: 't', arguments: '{}' },
              },
            ],
          },
          finish_reason: 'tool_calls',
        },
      ],
    }),
);

runChatModelContract(
  'anthropic',
  (fetch) => new AnthropicChatModel({ apiKey: 'k', fetch }),
  (fetch) => new AnthropicChatModel({ apiKey: 'k', fetch }),
  async () =>
    jsonResponse({
      content: [{ type: 'text', text: 'ok' }],
      stop_reason: 'end_turn',
    }),
  async () =>
    jsonResponse({
      content: [{ type: 'tool_use', id: 'c', name: 't', input: {} }],
      stop_reason: 'tool_use',
    }),
);

describe('Providers + ChatClient tool loop (scripted remains primary path)', () => {
  test('OpenAI mock + ToolCallingAdvisor still works via ChatClient', async () => {
    let turn = 0;
    const fetchImpl: FetchLike = async () => {
      turn += 1;
      if (turn === 1) {
        return jsonResponse({
          choices: [
            {
              message: {
                content: null,
                tool_calls: [
                  {
                    id: 'call_w',
                    type: 'function',
                    function: {
                      name: 'getWeather',
                      arguments: '{"city":"Yorktown"}',
                    },
                  },
                ],
              },
              finish_reason: 'tool_calls',
            },
          ],
        });
      }
      return jsonResponse({
        choices: [
          {
            message: { content: '68F in Yorktown' },
            finish_reason: 'stop',
          },
        ],
      });
    };

    const weather = functionToolCallback({
      name: 'getWeather',
      description: 'Get weather',
      inputSchema: {
        type: 'object',
        properties: { city: { type: 'string' } },
        required: ['city'],
      },
      call: ({ city }: { city: string }) => ({ temp: 68, city }),
    });

    const client = ChatClient.create(new OpenAiChatModel({ apiKey: 'sk', fetch: fetchImpl }));

    const answer = await client
      .prompt()
      .user('What is the weather in Yorktown?')
      .tools(weather)
      .call()
      .content();

    expect(answer).toBe('68F in Yorktown');
    expect(turn).toBe(2);
  });

  test('contract: ScriptedChatModel still satisfies portable call API', async () => {
    const model = new ScriptedChatModel([{ respond: 'portable' }]);
    const response = await model.call(new Prompt('q'));
    expect(response.content).toBe('portable');
  });
});
