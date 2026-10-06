import { describe, expect, test } from 'bun:test';
import {
  assistantMessage,
  systemMessage,
  toolCall,
  toolResponseMessage,
  userMessage,
} from '../src/chat/messages/factories.ts';
import { Prompt } from '../src/chat/prompt/prompt.ts';
import { media } from '../src/content/media.ts';
import { textDocument } from '../src/document/document.ts';
import {
  WorkersAiChatModel,
  WorkersAiEmbeddingModel,
  workersAiChatModel,
  workersAiEmbeddingModel,
} from '../src/provider/workers-ai/index.ts';
import { functionToolCallback } from '../src/tool/function-tool-callback.ts';

const modelId = '@cf/meta/llama-3.1-8b-instruct';

describe('WorkersAiChatModel', () => {
  test('calls the binding and maps a Workers AI response', async () => {
    let seenModel = '';
    let seenInputs: Record<string, unknown> = {};
    const binding = {
      async run(model: string, inputs: Record<string, unknown>, options?: { gateway?: unknown }) {
        seenModel = model;
        seenInputs = inputs;
        expect(options?.gateway).toEqual({ id: 'gw' });
        return {
          response: 'Yorktown',
          usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
        };
      },
    };
    const model = WorkersAiChatModel.of(
      { binding },
      {
        model: modelId,
        temperature: 0.2,
        gateway: { id: 'gw' },
      },
    );
    const response = await model.call(
      new Prompt(
        [
          systemMessage('Be brief.'),
          userMessage('Where?'),
          assistantMessage('', { toolCalls: [toolCall('c1', 'lookup', { q: 'town' })] }),
          toolResponseMessage([{ id: 'c1', name: 'lookup', responseData: '{"city":"Yorktown"}' }]),
        ],
        {
          toolCallbacks: [
            functionToolCallback({
              name: 'lookup',
              description: 'Look up a place',
              inputSchema: { type: 'object', properties: { q: { type: 'string' } } },
              call: () => 'ok',
            }),
          ],
          maxTokens: 32,
        },
      ),
    );
    expect(seenModel).toBe(modelId);
    expect(seenInputs.temperature).toBe(0.2);
    expect(seenInputs.max_tokens).toBe(32);
    const messages = seenInputs.messages as Array<Record<string, unknown>>;
    expect(messages[2]?.tool_calls).toEqual([{ name: 'lookup', arguments: { q: 'town' } }]);
    expect(messages[3]).toMatchObject({ role: 'tool', name: 'lookup' });
    expect(response.content).toBe('Yorktown');
    expect(response.metadata.usage?.totalTokens).toBe(4);
  });

  test('maps tool calls and an OpenAI-shaped payload', async () => {
    const model = new WorkersAiChatModel({
      binding: asyncBinding({
        choices: [
          {
            message: {
              content: null,
              tool_calls: [{ id: 'call_1', function: { name: 'ping', arguments: '{"ok":true}' } }],
            },
            finish_reason: 'tool_calls',
          },
        ],
      }),
    });
    const response = await model.call(new Prompt('hi'));
    expect(response.hasToolCalls()).toBe(true);
    expect(response.getResult()?.output.toolCalls[0]).toMatchObject({
      id: 'call_1',
      name: 'ping',
      arguments: '{"ok":true}',
    });
  });

  test('resolves a getter after the Worker env is published', async () => {
    let env: {
      AI?: { run: (model: string, inputs: Record<string, unknown>) => Promise<unknown> };
    } = {};
    const model = new WorkersAiChatModel({
      binding: () => env.AI,
      model: modelId,
    });
    await expect(model.call(new Prompt('hi'))).rejects.toMatchObject({ code: 'invalid-request' });
    env = { AI: asyncBinding({ response: 'ready' }) };
    expect((await model.call(new Prompt('hi'))).content).toBe('ready');
  });

  test('rejects an empty prompt, media, and provider failures', async () => {
    const model = new WorkersAiChatModel({
      binding: {
        run: async () => {
          throw new Error('upstream down');
        },
      },
    });
    await expect(model.call(new Prompt([]))).rejects.toMatchObject({ code: 'invalid-request' });
    await expect(
      model.call(new Prompt([userMessage('see', { media: [media('image/png', 'aGVsbG8=')] })])),
    ).rejects.toMatchObject({ code: 'invalid-request' });
    await expect(model.call(new Prompt('hi'))).rejects.toMatchObject({
      code: 'provider-error',
      message: 'upstream down',
    });
  });

  test('stops when the request is already cancelled', async () => {
    const controller = new AbortController();
    controller.abort();
    const model = new WorkersAiChatModel({ binding: asyncBinding({ response: 'nope' }) });
    await expect(model.call(new Prompt('hi', { signal: controller.signal }))).rejects.toMatchObject(
      {
        code: 'cancelled',
      },
    );
  });

  test('streams SSE token deltas and a terminal tool call', async () => {
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('data: {"response":"Hello"}\n\n'));
        controller.enqueue(encoder.encode('data: {"response":" world"}\n'));
        controller.enqueue(
          encoder.encode(
            'data: {"response":"","tool_calls":[{"name":"ping","arguments":{"ok":true}}]}\n',
          ),
        );
        controller.enqueue(encoder.encode('data: [DONE]\n'));
        controller.close();
      },
    });
    const model = new WorkersAiChatModel({ binding: asyncBinding(stream) });
    const chunks = [];
    for await (const chunk of model.stream(new Prompt('hi')) ?? []) {
      chunks.push(chunk);
    }
    expect(chunks.at(-1)?.content).toBe('Hello world');
    expect(chunks.at(-1)?.hasToolCalls()).toBe(true);
  });

  test('streams an async iterable and constructs a model from the factory', async () => {
    async function* events() {
      yield { response: 'one' };
      yield 'data: {"response":"two"}\n';
    }
    const model = workersAiChatModel({ binding: asyncBinding(events()) });
    const chunks = [];
    for await (const chunk of model.stream(new Prompt('hi')) ?? []) {
      chunks.push(chunk.content);
    }
    expect(chunks).toEqual(['one', 'onetwo']);
  });

  test('appends token deltas and replaces a full message snapshot', async () => {
    async function* events() {
      yield { response: 'ha' };
      yield { response: 'ha' };
      yield { choices: [{ delta: { content: ' ' } }] };
      yield { choices: [{ delta: { content: 'The' } }] };
      yield { choices: [{ message: { content: 'done' } }] };
    }
    const model = new WorkersAiChatModel({ binding: asyncBinding(events()) });
    const chunks = [];
    for await (const chunk of model.stream(new Prompt('hi')) ?? []) {
      chunks.push(chunk.content);
    }
    expect(chunks).toEqual(['ha', 'haha', 'haha ', 'haha The', 'done']);
  });
});

describe('WorkersAiEmbeddingModel', () => {
  test('embeds a batch from data rows and a wrapped flat vector', async () => {
    const rows = new WorkersAiEmbeddingModel({
      binding: asyncBinding({
        data: [
          [0.1, 0.2],
          [0.3, 0.4],
        ],
      }),
    });
    expect(await rows.embedBatch(['a', 'b'])).toEqual([
      [0.1, 0.2],
      [0.3, 0.4],
    ]);

    const flat = WorkersAiEmbeddingModel.of({
      binding: asyncBinding({ result: { shape: [1, 2], data: [0.5, 0.6] } }),
    });
    const doc = textDocument('hello', {}, 'd1');
    expect(await flat.embedDocument(doc)).toEqual([0.5, 0.6]);

    const factory = workersAiEmbeddingModel({
      binding: asyncBinding({ data: [[1, 2]] }),
    });
    expect(await factory.embed('x')).toEqual([1, 2]);
  });

  test('reports a provider error when the payload has no vectors', async () => {
    const model = new WorkersAiEmbeddingModel({ binding: asyncBinding({ data: 'nope' }) });
    await expect(model.embed('hi')).rejects.toMatchObject({ code: 'provider-error' });
    expect(await new WorkersAiEmbeddingModel({ binding: asyncBinding({}) }).embedBatch([])).toEqual(
      [],
    );
  });
});

function asyncBinding(payload: unknown) {
  return {
    run: async () => payload,
  };
}
