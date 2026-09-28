import { describe, expect, test } from 'bun:test';
import {
  assistantMessage,
  CHAT_MEMORY_CONVERSATION_ID,
  type ChatMessage,
  ContextCompressionAdvisor,
  ContextCompressionError,
  chatClientRequest,
  chatClientResponse,
  DEFAULT_CHAT_MEMORY_PRECEDENCE_ORDER,
  DEFAULT_CONTEXT_COMPRESSION_ORDER,
  HIGHEST_PRECEDENCE,
  MessageWindowChatMemory,
  media,
  Prompt,
  systemMessage,
  toolCall,
  toolResponse,
  toolResponseMessage,
  userMessage,
} from '../src/index.ts';

const counter = { count: (messages: readonly unknown[]) => messages.length * 10 };

function conversation() {
  return [
    systemMessage('system'),
    userMessage('old user'),
    assistantMessage('old assistant'),
    assistantMessage(null, { toolCalls: [toolCall('1', 'lookup', {})] }),
    toolResponseMessage([toolResponse('1', 'lookup', 'result')]),
    userMessage('image', { media: [media('image/png', 'data')] }),
    userMessage('current'),
  ];
}

describe('ContextCompressionAdvisor', () => {
  test('runs after memory and before retrieval while preserving protected messages', async () => {
    const original = conversation();
    const diagnostics: unknown[] = [];
    const advisor = new ContextCompressionAdvisor({
      tokenBudget: 50,
      tokenCounter: counter,
      compressor: {
        compress(request) {
          expect(request.protectedIndices).toEqual([0, 3, 4, 5, 6]);
          return {
            messages: request.protectedIndices.map(
              (index) => request.messages[index] as ChatMessage,
            ),
            compressedRanges: [{ start: 1, end: 3 }],
          };
        },
      },
      onCompression: (diagnostic) => diagnostics.push(diagnostic),
    });
    expect(DEFAULT_CONTEXT_COMPRESSION_ORDER).toBeGreaterThan(DEFAULT_CHAT_MEMORY_PRECEDENCE_ORDER);
    expect(DEFAULT_CONTEXT_COMPRESSION_ORDER).toBeLessThan(HIGHEST_PRECEDENCE + 250);
    const processed = await advisor.before(chatClientRequest(new Prompt(original)));
    expect(processed.prompt.messages.map((message) => message.text)).toEqual([
      'system',
      null,
      null,
      'image',
      'current',
    ]);
    expect(diagnostics).toMatchObject([
      {
        originalTokens: 70,
        finalTokens: 50,
        compressedRanges: [{ start: 1, end: 3 }],
        persistence: 'request',
      },
    ]);
  });

  test('rejects changed protected messages and still-over-budget output', async () => {
    const request = chatClientRequest(new Prompt(conversation()));
    const removing = new ContextCompressionAdvisor({
      tokenBudget: 50,
      tokenCounter: counter,
      compressor: { compress: () => ({ messages: [], compressedRanges: [{ start: 1, end: 3 }] }) },
    });
    await expect(removing.before(request)).rejects.toMatchObject({
      code: 'PROTECTED_MESSAGE_REMOVED',
    });

    const overBudget = new ContextCompressionAdvisor({
      tokenBudget: 50,
      tokenCounter: counter,
      compressor: {
        compress: () => ({ messages: conversation(), compressedRanges: [{ start: 1, end: 2 }] }),
      },
    });
    await expect(overBudget.before(request)).rejects.toMatchObject({ code: 'OVER_BUDGET' });
  });

  test('persists atomically only for replaceable memories', async () => {
    const plainMemory = {
      add() {},
      get() {
        return [];
      },
      clear() {},
    };
    expect(
      () =>
        new ContextCompressionAdvisor({
          tokenBudget: 10,
          tokenCounter: counter,
          compressor: { compress: () => ({ messages: [], compressedRanges: [] }) },
          persistence: 'memory',
          chatMemory: plainMemory,
        }),
    ).toThrow(ContextCompressionError);

    const memory = MessageWindowChatMemory.of({ maxMessages: 20 });
    memory.add('conversation', conversation());
    const advisor = new ContextCompressionAdvisor({
      tokenBudget: 50,
      tokenCounter: counter,
      compressor: {
        compress: (request) => ({
          messages: request.protectedIndices.map((index) => request.messages[index] as ChatMessage),
          compressedRanges: [{ start: 1, end: 3 }],
        }),
      },
      persistence: 'memory',
      chatMemory: memory,
    });
    const request = chatClientRequest(new Prompt(conversation()), {
      [CHAT_MEMORY_CONVERSATION_ID]: 'conversation',
    });
    await advisor.before(request);
    expect(memory.get('conversation').map((message) => message.text)).toEqual([
      'system',
      null,
      null,
      'image',
      'current',
    ]);
  });

  test('applies on both call and stream paths and skips within-budget requests', async () => {
    let compressions = 0;
    const advisor = new ContextCompressionAdvisor({
      tokenBudget: 10,
      tokenCounter: { count: () => 1 },
      compressor: {
        compress: () => {
          compressions++;
          return { messages: [], compressedRanges: [] };
        },
      },
    });
    const request = chatClientRequest(new Prompt([userMessage('current')]));
    await advisor.adviseCall(request, {
      callAdvisors: [],
      nextCall: async (next) => chatClientResponse(undefined, next.context),
    });
    const streamed = [];
    for await (const response of advisor.adviseStream(request, {
      streamAdvisors: [],
      async *nextStream(next) {
        yield chatClientResponse(undefined, next.context);
      },
    }))
      streamed.push(response);
    expect(streamed).toHaveLength(1);
    expect(compressions).toBe(0);
  });
});
