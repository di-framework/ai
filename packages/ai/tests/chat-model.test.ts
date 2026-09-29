import { describe, expect, test } from 'bun:test';
import {
  AiError,
  assistantMessage,
  ChatResponse,
  FakeChatModel,
  Prompt,
  RecordingChatModel,
  requestContains,
  ScriptedChatModel,
  textResponse,
  toolCall,
  toolCallResponse,
  userMessage,
} from '../src/index.ts';

describe('FakeChatModel', () => {
  test('returns fixed text', async () => {
    const model = new FakeChatModel('pong');
    const response = await model.call(new Prompt('ping'));
    expect(response.content).toBe('pong');
    expect(model.calls).toHaveLength(1);
  });

  test('handler can inspect messages', async () => {
    const model = new FakeChatModel((prompt) => {
      const last = prompt.getUserMessage().text ?? '';
      return ChatResponse.of(`echo:${last}`);
    });
    const response = await model.call(Prompt.fromMessages([userMessage('Yorktown')]));
    expect(response.content).toBe('echo:Yorktown');
  });

  test('respects AbortSignal', async () => {
    const controller = new AbortController();
    controller.abort();
    const model = new FakeChatModel('x');
    await expect(
      model.call(new Prompt('hi', { signal: controller.signal })),
    ).rejects.toBeInstanceOf(AiError);
  });

  test('stream yields progressive text', async () => {
    const model = new FakeChatModel('a b');
    const chunks: string[] = [];
    for await (const part of model.stream(new Prompt('q'))) {
      chunks.push(part.content);
    }
    expect(chunks.at(-1)).toBe('a b');
    expect(chunks.length).toBeGreaterThan(1);
  });
});

describe('ScriptedChatModel', () => {
  test('walks tool call then final answer', async () => {
    const model = new ScriptedChatModel([
      {
        when: requestContains('weather'),
        respond: toolCallResponse([toolCall('1', 'get-weather', { location: 'Yorktown' })]),
      },
      {
        respond: textResponse('It is 72°F.'),
      },
    ]);

    const first = await model.call(new Prompt('What is the weather?'));
    expect(first.hasToolCalls()).toBe(true);
    expect(first.getResult()?.output.toolCalls[0]?.name).toBe('get-weather');

    const second = await model.call(new Prompt('tool result context'));
    expect(second.content).toBe('It is 72°F.');
  });

  test('throws when script is exhausted', async () => {
    const model = new ScriptedChatModel([{ respond: 'only' }]);
    await model.call(new Prompt('1'));
    await expect(model.call(new Prompt('2'))).rejects.toBeInstanceOf(AiError);
  });
});

describe('ChatResponse', () => {
  test('hasToolCalls detects assistant tool calls', () => {
    const response = ChatResponse.fromAssistant(
      assistantMessage(null, {
        toolCalls: [toolCall('c1', 'search', { q: 'x' })],
      }),
    );
    expect(response.hasToolCalls()).toBe(true);
    expect(response.content).toBe('');
  });
});

describe('RecordingChatModel', () => {
  test('records delegated calls', async () => {
    const inner = new FakeChatModel('ok');
    const model = new RecordingChatModel(inner);
    await model.call(new Prompt('a'));
    await model.call(new Prompt('b'));
    expect(model.calls).toHaveLength(2);
    expect(inner.calls).toHaveLength(2);
  });

  test('stream() delegates to the wrapped model and records the prompt', async () => {
    const inner = new FakeChatModel('a b');
    const model = new RecordingChatModel(inner);
    const chunks: string[] = [];
    for await (const response of model.stream(new Prompt('go'))) {
      if (response.content) chunks.push(response.content);
    }
    expect(chunks.at(-1)).toBe('a b');
    expect(model.calls).toHaveLength(1);
  });

  test('stream() throws when the delegate does not support streaming', () => {
    const inner = { call: async () => textResponse('ok') };
    const model = new RecordingChatModel(inner);
    expect(() => model.stream(new Prompt('go'))).toThrow(/does not support streaming/);
  });
});
