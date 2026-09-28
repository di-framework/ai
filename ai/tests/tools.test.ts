import { describe, expect, test } from 'bun:test';
import {
  type CallAdvisor,
  type CallAdvisorChain,
  ChatClient,
  type ChatClientRequest,
  createToolCallingManager,
  DEFAULT_TOOL_CALLING_ORDER,
  defaultToolExecutionExceptionProcessor,
  FunctionToolCallback,
  functionToolCallback,
  hasToolCalls,
  isToolCallback,
  isToolCallbackProvider,
  isToolResponseMessage,
  Prompt,
  resolveToolCallbacks,
  ScriptedChatModel,
  staticToolCallbackProvider,
  ToolCallingAdvisor,
  ToolContext,
  ToolExecutionException,
  textResponse,
  toolCall,
  toolCallResponse,
  toolDefinition,
  userMessage,
} from '../src/index.ts';
import { defaultToolCallResultConverter } from '../src/tool/execution/tool-call-result-converter.ts';
import { resolveToolCallbacksDedupe } from '../src/tool/tool-callback-provider.ts';

describe('ToolDefinition / FunctionToolCallback', () => {
  test('toolDefinition defaults description and schema', () => {
    const def = toolDefinition({ name: 'getWeather' });
    expect(def.name).toBe('getWeather');
    expect(def.description).toBe('get Weather');
    expect(def.inputSchema).toContain('object');
  });

  test('toolDefinition throws for an empty/whitespace name', () => {
    expect(() => toolDefinition({ name: '   ' })).toThrow(/name cannot be empty/);
  });

  test('toolDefinition falls back to the default schema for a blank string schema', () => {
    const def = toolDefinition({ name: 'ping', inputSchema: '   ' });
    expect(def.inputSchema).toBe('{"type":"object","properties":{}}');
  });

  test('toolDefinition keeps a trimmed string schema as-is', () => {
    const def = toolDefinition({ name: 'ping', inputSchema: '  {"type":"string"}  ' });
    expect(def.inputSchema).toBe('{"type":"string"}');
  });

  test('defaultToolCallResultConverter falls back to String() when JSON.stringify throws', () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    expect(defaultToolCallResultConverter(circular)).toBe(String(circular));
    expect(defaultToolCallResultConverter(null)).toBe('');
    expect(defaultToolCallResultConverter('already a string')).toBe('already a string');
  });

  test('functionToolCallback parses JSON and returns stringified result', async () => {
    const cb = functionToolCallback<{ city: string }, { temp: number }>({
      name: 'getWeather',
      description: 'weather lookup',
      inputSchema: {
        type: 'object',
        properties: { city: { type: 'string' } },
        required: ['city'],
      },
      call: ({ city }) => ({ temp: 72, city }),
    });

    const result = await cb.call('{"city":"Yorktown"}');
    expect(JSON.parse(result)).toEqual({ temp: 72, city: 'Yorktown' });
    expect(cb.toolDefinition.name).toBe('getWeather');
  });

  test('functionToolCallback wraps execution errors', async () => {
    const cb = functionToolCallback({
      name: 'boom',
      call: () => {
        throw new Error('noop');
      },
    });

    await expect(cb.call('{}')).rejects.toBeInstanceOf(ToolExecutionException);
  });

  test('tool context is passed through', async () => {
    let seen: unknown;
    const cb = functionToolCallback({
      name: 'ctx',
      call: (_input, context) => {
        seen = context?.get('userId');
        return 'ok';
      },
    });

    await cb.call('{}', new ToolContext({ userId: 'u-1' }));
    expect(seen).toBe('u-1');
  });

  test('defaultToolExecutionExceptionProcessor rethrows by default and can be configured to return a JSON error string', async () => {
    const cb = functionToolCallback({
      name: 'boom',
      call: () => {
        throw new Error('noop');
      },
    });
    let caught: unknown;
    try {
      await cb.call('{}');
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(ToolExecutionException);
    const error = caught as ToolExecutionException;

    const alwaysThrowProcessor = defaultToolExecutionExceptionProcessor();
    expect(() => alwaysThrowProcessor(error)).toThrow(error);

    const returnErrorProcessor = defaultToolExecutionExceptionProcessor({ alwaysThrow: false });
    const payload = JSON.parse(returnErrorProcessor(error));
    expect(payload).toEqual({ error: true, message: error.message, tool: 'boom' });
  });

  test('staticToolCallbackProvider exposes callbacks', () => {
    const a = functionToolCallback({ name: 'a', call: () => 'a' });
    const provider = staticToolCallbackProvider([a]);
    expect(provider.getToolCallbacks()).toHaveLength(1);
  });

  test('resolveToolCallbacks flattens arrays, providers, and single callbacks', () => {
    const a = functionToolCallback({ name: 'a', call: () => 'a' });
    const b = functionToolCallback({ name: 'b', call: () => 'b' });
    const c = functionToolCallback({ name: 'c', call: () => 'c' });
    const provider = staticToolCallbackProvider([b]);

    const result = resolveToolCallbacks([a], provider, c, null, undefined);
    expect(result.map((cb) => cb.toolDefinition.name)).toEqual(['a', 'b', 'c']);
  });

  test('resolveToolCallbacks rejects unrecognized source values', () => {
    expect(() => resolveToolCallbacks(42 as never)).toThrow(
      /Expected ToolCallback, ToolCallbackProvider, or array/,
    );
  });

  test('resolveToolCallbacks throws on duplicate tool names', () => {
    const a = functionToolCallback({ name: 'dup', call: () => 'a' });
    const b = functionToolCallback({ name: 'dup', call: () => 'b' });
    expect(() => resolveToolCallbacks([a, b])).toThrow(/Multiple tools with the same name/);
  });

  test('isToolCallback / isToolCallbackProvider distinguish shapes', () => {
    const cb = functionToolCallback({ name: 'x', call: () => 'x' });
    const provider = staticToolCallbackProvider([cb]);
    expect(isToolCallback(cb)).toBe(true);
    expect(isToolCallback(provider)).toBe(false);
    expect(isToolCallback(null)).toBe(false);
    expect(isToolCallbackProvider(provider)).toBe(true);
    expect(isToolCallbackProvider(cb)).toBe(false);
    expect(isToolCallbackProvider(null)).toBe(false);
  });

  test('resolveToolCallbacksDedupe keeps the last callback per tool name across arrays/providers', () => {
    const aFirst = functionToolCallback({ name: 'a', call: () => 'first' });
    const aSecond = functionToolCallback({ name: 'a', call: () => 'second' });
    const provider = staticToolCallbackProvider([aSecond]);

    const result = resolveToolCallbacksDedupe([aFirst], provider);
    expect(result).toHaveLength(1);
    expect(result[0]).toBe(aSecond);
  });

  test('resolveToolCallbacksDedupe rejects unrecognized source values', () => {
    expect(() => resolveToolCallbacksDedupe(42 as never)).toThrow(
      /Expected ToolCallback, ToolCallbackProvider, or array/,
    );
  });
});

describe('ToolContext.has / toRecord', () => {
  test('has() reports whether a key is present', () => {
    const ctx = new ToolContext({ userId: 'u-1' });
    expect(ctx.has('userId')).toBe(true);
    expect(ctx.has('missing')).toBe(false);
  });

  test('toRecord() converts the context back to a plain object', () => {
    const ctx = new ToolContext(
      new Map([
        ['userId', 'u-1'],
        ['tenant', 't1'],
      ]),
    );
    expect(ctx.toRecord()).toEqual({ userId: 'u-1', tenant: 't1' });
  });
});

describe('ToolCallingManager', () => {
  test('executeToolCalls appends assistant + tool response messages', async () => {
    const weather = functionToolCallback<{ city: string }>({
      name: 'getWeather',
      call: ({ city }) => `sunny in ${city}`,
    });

    const manager = createToolCallingManager();
    const prompt = Prompt.fromMessages([userMessage('weather?')], {
      toolCallbacks: [weather],
    });

    const response = toolCallResponse([toolCall('call-1', 'getWeather', { city: 'Yorktown' })]);

    const result = await manager.executeToolCalls(prompt, response);
    expect(result.returnDirect).toBe(false);
    expect(result.conversationHistory).toHaveLength(3);
    expect(result.conversationHistory[1]?.messageType).toBe('assistant');
    expect(result.conversationHistory[2]?.messageType).toBe('tool');

    const toolMsg = result.conversationHistory[2]!;
    expect(isToolResponseMessage(toolMsg)).toBe(true);
    if (isToolResponseMessage(toolMsg)) {
      expect(toolMsg.responses[0]?.responseData).toBe('sunny in Yorktown');
      expect(toolMsg.responses[0]?.id).toBe('call-1');
    }
  });

  test('returnDirect is true when metadata says so', async () => {
    const cb = functionToolCallback({
      name: 'direct',
      returnDirect: true,
      call: () => 'payload',
    });

    const manager = createToolCallingManager();
    const prompt = new Prompt('hi', { toolCallbacks: [cb] });
    const response = toolCallResponse([toolCall('1', 'direct', {})]);

    const result = await manager.executeToolCalls(prompt, response);
    expect(result.returnDirect).toBe(true);
  });

  test('resolveToolDefinitions maps callbacks', () => {
    const cb = functionToolCallback({ name: 'x', description: 'X', call: () => '' });
    const defs = createToolCallingManager().resolveToolDefinitions({
      toolCallbacks: [cb],
    });
    expect(defs).toEqual([cb.toolDefinition]);
  });
});

describe('ChatClient tools + ToolCallingAdvisor', () => {
  test('end-to-end tool loop with ScriptedChatModel', async () => {
    const weather = functionToolCallback<{ city: string }>({
      name: 'getWeather',
      description: 'Get weather for a city',
      inputSchema: {
        type: 'object',
        properties: { city: { type: 'string' } },
        required: ['city'],
      },
      call: ({ city }) => ({ city, tempF: 68, conditions: 'clear' }),
    });

    const model = new ScriptedChatModel([
      {
        respond: toolCallResponse([toolCall('c1', 'getWeather', { city: 'Yorktown' })]),
      },
      {
        when: (prompt) =>
          prompt.messages.some(
            (m) =>
              isToolResponseMessage(m) &&
              m.responses.some((r) => r.responseData.includes('Yorktown')),
          ),
        respond: textResponse('It is 68°F and clear in Yorktown.'),
      },
    ]);

    const client = ChatClient.create(model);

    const answer = await client
      .prompt()
      .system('Answer weather questions concisely.')
      .user('What is the weather in Yorktown?')
      .tools(weather)
      .call()
      .content();

    expect(answer).toBe('It is 68°F and clear in Yorktown.');
    expect(model.calls).toHaveLength(2);

    // Second call should include tool response history
    const second = model.calls[1]!;
    expect(second.messages.some(isToolResponseMessage)).toBe(true);
    expect(second.options?.toolCallbacks).toHaveLength(1);
  });

  test('tools appear on toRequest options', () => {
    const cb = functionToolCallback({ name: 'ping', call: () => 'pong' });
    const request = ChatClient.create(new ScriptedChatModel([]))
      .prompt()
      .user('hi')
      .tools(cb)
      .toolContext({ tenant: 't1' })
      .toRequest();

    expect(request.prompt.options?.toolCallbacks).toHaveLength(1);
    expect(request.prompt.options?.toolContext).toEqual({ tenant: 't1' });
  });

  test('builder defaultTools apply when call tools omitted', async () => {
    const calls: string[] = [];
    const cb = functionToolCallback({
      name: 'echo',
      call: (input: { text: string }) => {
        calls.push(input.text);
        return input.text;
      },
    });

    const model = new ScriptedChatModel([
      {
        respond: toolCallResponse([toolCall('1', 'echo', { text: 'hello' })]),
      },
      { respond: textResponse('done') },
    ]);

    const client = ChatClient.builder(model).defaultTools(cb).build();
    const content = await client.prompt().user('q').call().content();
    expect(content).toBe('done');
    expect(calls).toEqual(['hello']);
  });

  test('returnDirect short-circuits the model loop', async () => {
    const cb = functionToolCallback({
      name: 'lookup',
      returnDirect: true,
      call: () => 'direct-result',
    });

    const model = new ScriptedChatModel([
      {
        respond: toolCallResponse([toolCall('1', 'lookup', {})]),
      },
      { respond: textResponse('should-not-run') },
    ]);

    const content = await ChatClient.create(model).prompt().user('q').tools(cb).call().content();

    expect(content).toBe('direct-result');
    expect(model.calls).toHaveLength(1);
  });

  test('explicit ToolCallingAdvisor is not double-registered', async () => {
    let toolAdvisorPasses = 0;
    const counter: CallAdvisor = {
      name: 'counter',
      order: DEFAULT_TOOL_CALLING_ORDER + 100,
      async adviseCall(request: ChatClientRequest, chain: CallAdvisorChain) {
        toolAdvisorPasses += 1;
        return chain.nextCall(request);
      },
    };

    const model = new ScriptedChatModel([
      {
        respond: toolCallResponse([toolCall('1', 'noop', {}, 'function')]),
      },
      { respond: textResponse('final') },
    ]);

    // noop tool
    const noop = functionToolCallback({ name: 'noop', call: () => 'ok' });

    await ChatClient.create(model)
      .prompt()
      .user('q')
      .tools(noop)
      .advisors(ToolCallingAdvisor.builder().build(), counter)
      .call()
      .content();

    // counter sits after tool advisor: once per model round (tool call + final)
    expect(toolAdvisorPasses).toBe(2);
  });

  test('advisors after ToolCallingAdvisor see every tool iteration', async () => {
    const rounds: number[] = [];
    let n = 0;
    const counter: CallAdvisor = {
      name: 'round-counter',
      order: DEFAULT_TOOL_CALLING_ORDER + 50,
      async adviseCall(request, chain) {
        n += 1;
        rounds.push(request.prompt.messages.length);
        return chain.nextCall(request);
      },
    };

    const tool = functionToolCallback({
      name: 'step',
      call: () => 'step-ok',
    });

    const model = new ScriptedChatModel([
      { respond: toolCallResponse([toolCall('a', 'step', {})]) },
      { respond: textResponse('all done') },
    ]);

    await ChatClient.create(model)
      .prompt()
      .user('go')
      .tools(tool)
      .advisors(counter)
      .call()
      .content();

    expect(n).toBe(2);
    // second round has more messages (history + tool responses)
    expect(rounds[1]!).toBeGreaterThan(rounds[0]!);
  });

  test('stream path completes tool loop and yields final answer', async () => {
    const tool = functionToolCallback({
      name: 'add',
      call: (input: { a: number; b: number }) => String(input.a + input.b),
    });

    const model = new ScriptedChatModel([
      {
        respond: toolCallResponse([toolCall('1', 'add', { a: 2, b: 3 })]),
      },
      { respond: textResponse('5') },
    ]);

    // Fake stream via FakeChatModel-like: ScriptedChatModel has no stream.
    // Use a thin wrapper.
    const streamingModel = {
      options: model.options,
      call: (p: Prompt) => model.call(p),
      async *stream(prompt: Prompt) {
        const response = await model.call(prompt);
        yield response;
      },
    };

    const chunks: string[] = [];
    for await (const part of ChatClient.create(streamingModel)
      .prompt()
      .user('2+3?')
      .tools(tool)
      .stream()
      .content()) {
      chunks.push(part);
    }

    expect(chunks.at(-1)).toBe('5');
    expect(model.calls).toHaveLength(2);
  });

  test('without tools, ToolCallingAdvisor is a no-op pass-through', async () => {
    const model = new ScriptedChatModel([{ respond: textResponse('plain') }]);
    const content = await ChatClient.create(model).prompt().user('hi').call().content();
    expect(content).toBe('plain');
    expect(model.calls).toHaveLength(1);
  });

  test('FunctionToolCallback class export works', () => {
    const cb = new FunctionToolCallback({
      name: 'klass',
      call: () => 1,
    });
    expect(cb.toolDefinition.name).toBe('klass');
  });
});

describe('hasToolCalls helper', () => {
  test('detects tool calls on assistant messages', () => {
    const response = toolCallResponse([toolCall('1', 'x', {})]);
    expect(response.hasToolCalls()).toBe(true);
    const output = response.getResult()?.output;
    expect(output).toBeDefined();
    expect(hasToolCalls(output!)).toBe(true);
  });
});

describe('ToolCallingAdvisorBuilder fluent setters', () => {
  test('each setter configures the built advisor', async () => {
    const manager = createToolCallingManager();
    let checked = 0;
    const advisor = ToolCallingAdvisor.builder()
      .toolCallingManager(manager)
      .toolExecutionEligibilityChecker((response) => {
        checked += 1;
        const output = response?.getResult()?.output;
        return output != null && hasToolCalls(output);
      })
      .advisorOrder(DEFAULT_TOOL_CALLING_ORDER + 1)
      .conversationHistoryEnabled(true)
      .build();

    expect(advisor.order).toBe(DEFAULT_TOOL_CALLING_ORDER + 1);

    const model = new ScriptedChatModel([{ respond: textResponse('final') }]);
    const tool = functionToolCallback({ name: 'noop', call: () => 'ok' });
    const content = await ChatClient.create(model)
      .prompt()
      .user('q')
      .tools(tool)
      .advisors(advisor)
      .call()
      .content();

    expect(content).toBe('final');
    expect(checked).toBeGreaterThan(0);
  });

  test('disableInternalConversationHistory() sets conversationHistoryEnabled to false', async () => {
    const advisor = ToolCallingAdvisor.builder().disableInternalConversationHistory().build();
    const model = new ScriptedChatModel([
      { respond: toolCallResponse([toolCall('1', 'noop', {}, 'function')]) },
      { respond: textResponse('final') },
    ]);
    const tool = functionToolCallback({ name: 'noop', call: () => 'ok' });
    const content = await ChatClient.create(model)
      .prompt()
      .user('q')
      .tools(tool)
      .advisors(advisor)
      .call()
      .content();
    expect(content).toBe('final');
  });
});
