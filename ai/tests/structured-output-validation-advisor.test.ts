import { describe, expect, test } from 'bun:test';
import { StructuredOutputValidationAdvisor } from '../src/chat/client/advisor/structured-output-validation-advisor.ts';
import { chatClientRequest } from '../src/chat/client/chat-client-request.ts';
import { chatClientResponse } from '../src/chat/client/chat-client-response.ts';
import { assistantMessage } from '../src/chat/messages/factories.ts';
import { chatResponseMetadata } from '../src/chat/metadata/chat-response-metadata.ts';
import { ChatResponse } from '../src/chat/model/chat-response.ts';
import { generation } from '../src/chat/model/generation.ts';
import { Prompt } from '../src/chat/prompt/prompt.ts';
import type { CallAdvisor, CallAdvisorChain, StreamAdvisorChain } from '../src/index.ts';

const schema = JSON.stringify({
  type: 'object',
  properties: { name: { type: 'string' } },
  required: ['name'],
});

function toolCallResponse() {
  return new ChatResponse(
    [
      generation(
        assistantMessage(null, {
          toolCalls: [{ id: 't1', type: 'function', name: 'x', arguments: '{}' }],
        }),
        {
          finishReason: 'tool_calls',
        },
      ),
    ],
    chatResponseMetadata(),
  );
}

describe('ChatResponse', () => {
  test('content getter reads first result text', () => {
    const response = ChatResponse.of('hello');
    expect(response.content).toBe('hello');
  });

  test('result getter is an alias for getResult()', () => {
    const response = ChatResponse.of('hello');
    expect(response.result).toBe(response.getResult());
  });

  test('hasFinishReasons matches case-insensitively against a Set', () => {
    const response = new ChatResponse(
      [generation(assistantMessage('x'), { finishReason: 'STOP' })],
      chatResponseMetadata(),
    );
    expect(response.hasFinishReasons(new Set(['stop']))).toBe(true);
    expect(response.hasFinishReasons(['stop'])).toBe(true);
    expect(response.hasFinishReasons(['tool_calls'])).toBe(false);
  });
});

describe('StructuredOutputValidationAdvisor', () => {
  test('static builder creates an instance', () => {
    const advisor = StructuredOutputValidationAdvisor.builder({ outputJsonSchema: schema });
    expect(advisor).toBeInstanceOf(StructuredOutputValidationAdvisor);
  });

  test('throws when outputJsonSchema is empty', () => {
    expect(() => new StructuredOutputValidationAdvisor({ outputJsonSchema: '  ' })).toThrow(
      'outputJsonSchema must not be empty',
    );
  });

  test('throws when maxRepeatAttempts is negative', () => {
    expect(
      () =>
        new StructuredOutputValidationAdvisor({ outputJsonSchema: schema, maxRepeatAttempts: -1 }),
    ).toThrow('maxRepeatAttempts must be >= 0');
  });

  test('adviseStream passes through without validation', async () => {
    const advisor = new StructuredOutputValidationAdvisor({ outputJsonSchema: schema });
    const chain: StreamAdvisorChain = {
      streamAdvisors: [],
      async *nextStream() {
        yield chatClientResponse(ChatResponse.of('{"name":"x"}'));
      },
    };
    const results = [];
    for await (const r of advisor.adviseStream(chatClientRequest(new Prompt('q')), chain)) {
      results.push(r);
    }
    expect(results).toHaveLength(1);
  });

  test('succeeds immediately for tool-call responses without validating', async () => {
    const advisor = new StructuredOutputValidationAdvisor({ outputJsonSchema: schema });
    let calls = 0;
    const chain: CallAdvisorChain = {
      callAdvisors: [],
      async nextCall() {
        calls++;
        return chatClientResponse(toolCallResponse());
      },
    };
    const result = await advisor.adviseCall(chatClientRequest(new Prompt('q')), chain);
    expect(calls).toBe(1);
    expect(result.chatResponse?.hasToolCalls()).toBe(true);
  });

  test('uses chain.copy(this) when the chain supports it', async () => {
    const advisor = new StructuredOutputValidationAdvisor({ outputJsonSchema: schema });
    let copiedWith: CallAdvisor | undefined;
    let calls = 0;
    const innerChain: CallAdvisorChain = {
      callAdvisors: [],
      async nextCall() {
        calls++;
        return chatClientResponse(ChatResponse.of('{"name":"ok"}'));
      },
    };
    const chain: CallAdvisorChain = {
      callAdvisors: [],
      async nextCall() {
        throw new Error('should not be called directly');
      },
      copy(after) {
        copiedWith = after;
        return innerChain;
      },
    };
    const result = await advisor.adviseCall(chatClientRequest(new Prompt('q')), chain);
    expect(copiedWith).toBe(advisor);
    expect(calls).toBe(1);
    expect(result.chatResponse?.content).toBe('{"name":"ok"}');
  });

  test('re-prompts with the validation error appended to the user message on failure', async () => {
    const advisor = new StructuredOutputValidationAdvisor({
      outputJsonSchema: schema,
      maxRepeatAttempts: 1,
    });
    const seenPrompts: string[] = [];
    const chain: CallAdvisorChain = {
      callAdvisors: [],
      async nextCall(request) {
        seenPrompts.push(request.prompt.getUserMessage().text ?? '');
        if (seenPrompts.length === 1) {
          return chatClientResponse(ChatResponse.of('not-json'));
        }
        return chatClientResponse(ChatResponse.of('{"name":"fixed"}'));
      },
    };
    const result = await advisor.adviseCall(
      chatClientRequest(new Prompt('describe a person')),
      chain,
    );
    expect(seenPrompts).toHaveLength(2);
    expect(seenPrompts[1]).toContain('describe a person');
    expect(seenPrompts[1]).toContain('Output JSON validation failed');
    expect(result.chatResponse?.content).toBe('{"name":"fixed"}');
  });

  test('re-prompts using just the error message when there is no prior user text', async () => {
    const advisor = new StructuredOutputValidationAdvisor({
      outputJsonSchema: schema,
      maxRepeatAttempts: 1,
    });
    const seenPrompts: string[] = [];
    const chain: CallAdvisorChain = {
      callAdvisors: [],
      async nextCall(request) {
        seenPrompts.push(request.prompt.getUserMessage()?.text ?? '');
        if (seenPrompts.length === 1) {
          return chatClientResponse(ChatResponse.of('not-json'));
        }
        return chatClientResponse(ChatResponse.of('{"name":"fixed"}'));
      },
    };
    const result = await advisor.adviseCall(chatClientRequest(new Prompt([])), chain);
    expect(seenPrompts[1]).toContain('Output JSON validation failed');
    expect(result.chatResponse?.content).toBe('{"name":"fixed"}');
  });

  test('gives up after exhausting maxRepeatAttempts and returns the last (invalid) response', async () => {
    const advisor = new StructuredOutputValidationAdvisor({
      outputJsonSchema: schema,
      maxRepeatAttempts: 1,
    });
    let calls = 0;
    const chain: CallAdvisorChain = {
      callAdvisors: [],
      async nextCall() {
        calls++;
        return chatClientResponse(ChatResponse.of('still-not-json'));
      },
    };
    const result = await advisor.adviseCall(chatClientRequest(new Prompt('x')), chain);
    expect(calls).toBe(2);
    expect(result.chatResponse?.content).toBe('still-not-json');
  });

  test('reports missing output text', async () => {
    const advisor = new StructuredOutputValidationAdvisor({
      outputJsonSchema: schema,
      maxRepeatAttempts: 0,
    });
    const chain: CallAdvisorChain = {
      callAdvisors: [],
      async nextCall() {
        return chatClientResponse(undefined);
      },
    };
    const result = await advisor.adviseCall(chatClientRequest(new Prompt('x')), chain);
    expect(result.chatResponse).toBeUndefined();
  });

  test('uses a custom textCleaner', async () => {
    const advisor = new StructuredOutputValidationAdvisor({
      outputJsonSchema: schema,
      textCleaner: (text) => text.replace(/^WRAP\((.*)\)$/, '$1'),
    });
    const chain: CallAdvisorChain = {
      callAdvisors: [],
      async nextCall() {
        return chatClientResponse(ChatResponse.of('WRAP({"name":"cleaned"})'));
      },
    };
    const result = await advisor.adviseCall(chatClientRequest(new Prompt('x')), chain);
    expect(result.chatResponse?.content).toBe('WRAP({"name":"cleaned"})');
  });
});
