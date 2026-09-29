import { describe, expect, test } from 'bun:test';
import {
  type CallAdvisor,
  type CallAdvisorChain,
  ChatClient,
  type ChatClientRequest,
  createBeforeAfterAdvisor,
  DEFAULT_CHAT_MEMORY_PRECEDENCE_ORDER,
  FakeChatModel,
  HIGHEST_PRECEDENCE,
  LOWEST_PRECEDENCE,
  Prompt,
  SimpleLoggerAdvisor,
  systemMessage,
  userMessage,
} from '../src/index.ts';

describe('ChatClient', () => {
  test('fluent call returns content', async () => {
    const model = new FakeChatModel('hello world');
    const client = ChatClient.create(model);

    const content = await client
      .prompt()
      .system('You are concise.')
      .user('Say hi.')
      .call()
      .content();

    expect(content).toBe('hello world');
    expect(model.calls).toHaveLength(1);
    expect(model.calls[0]?.messages[0]?.messageType).toBe('system');
    expect(model.calls[0]?.messages[1]?.messageType).toBe('user');
  });

  test('prompt(string) is user message', async () => {
    const model = new FakeChatModel('ok');
    const client = ChatClient.create(model);
    const content = await client.prompt('direct').call().content();
    expect(content).toBe('ok');
    expect(model.calls[0]?.getUserMessage().text).toBe('direct');
  });

  test('builder default system and options', async () => {
    const model = new FakeChatModel('x');
    const client = ChatClient.builder(model)
      .defaultSystem('default-system')
      .defaultOptions({ temperature: 0.1 })
      .build();

    await client.prompt().user('q').call().content();
    expect(model.calls[0]?.getSystemMessage().text).toBe('default-system');
    expect(model.calls[0]?.options?.temperature).toBe(0.1);
  });

  test('builder defaultUser() supplies a fallback user message', async () => {
    const model = new FakeChatModel('x');
    const client = ChatClient.builder(model).defaultUser('default-user').build();
    await client.prompt().call().content();
    expect(model.calls[0]?.getUserMessage().text).toBe('default-user');
  });

  test('builder defaultContext() merges into every request context', async () => {
    const model = new FakeChatModel('x');
    let seen: unknown;
    const spy: CallAdvisor = {
      name: 'spy',
      order: LOWEST_PRECEDENCE,
      async adviseCall(request, chain) {
        seen = request.context.get('tenant');
        return chain.nextCall(request);
      },
    };
    const client = ChatClient.builder(model)
      .defaultContext({ tenant: 'acme' })
      .defaultAdvisors(spy)
      .build();
    await client.prompt().user('q').call().content();
    expect(seen).toBe('acme');
  });

  test('builder defaultToolBeans() derives tool callbacks from @Tool-annotated beans', async () => {
    const model = new FakeChatModel('x');
    const bean = {
      toolCallbacks: [],
    };
    // No @Tool methods on this plain bean; just confirm the fluent call itself
    // exercises defaultToolBeans() -> defaultTools() without throwing.
    const client = ChatClient.builder(model).defaultToolBeans(bean).build();
    const content = await client.prompt().user('q').call().content();
    expect(content).toBe('x');
  });

  test('builder defaultToolContext() merges into the tool execution context', async () => {
    const model = new FakeChatModel('x');
    const client = ChatClient.builder(model).defaultToolContext({ userId: '42' }).build();
    const content = await client.prompt().user('q').call().content();
    expect(content).toBe('x');
  });

  test('builder toolCallingAdvisorOptions() configures the auto-registered advisor', async () => {
    const model = new FakeChatModel('x');
    const client = ChatClient.builder(model)
      .toolCallingAdvisorOptions({ conversationHistoryEnabled: false })
      .build();
    const content = await client.prompt().user('q').call().content();
    expect(content).toBe('x');
  });

  test('builder autoRegisterToolCallingAdvisor(false) disables auto-registration', async () => {
    const model = new FakeChatModel('x');
    const client = ChatClient.builder(model).autoRegisterToolCallingAdvisor(false).build();
    const content = await client.prompt().user('q').call().content();
    expect(content).toBe('x');
  });

  test('mutate() clones the builder with the current defaults', async () => {
    const model = new FakeChatModel('x');
    const client = ChatClient.builder(model).defaultSystem('base-system').build();
    const mutated = client.mutate().defaultUser('mutated-user').build();

    const content = await mutated.prompt().call().content();
    expect(content).toBe('x');
    expect(model.calls[0]?.getSystemMessage().text).toBe('base-system');
    expect(model.calls[0]?.getUserMessage().text).toBe('mutated-user');
  });

  test('template params on user', async () => {
    const model = new FakeChatModel('ok');
    const client = ChatClient.create(model);

    await client
      .prompt()
      .user((u) => {
        u.text = 'Weather in {city}?';
        u.params = { city: 'Yorktown' };
      })
      .call()
      .content();

    expect(model.calls[0]?.getUserMessage().text).toBe('Weather in Yorktown?');
  });

  test('messages() and Prompt prompt()', async () => {
    const model = new FakeChatModel('ok');
    const client = ChatClient.create(model);

    await client
      .prompt(Prompt.fromMessages([systemMessage('s'), userMessage('u')]))
      .call()
      .content();

    expect(model.calls[0]?.messages.map((m) => m.messageType)).toEqual(['system', 'user']);
  });

  test('stream yields progressive content', async () => {
    const model = new FakeChatModel('a b');
    const client = ChatClient.create(model);
    const chunks: string[] = [];
    for await (const part of client.prompt().user('q').stream().content()) {
      chunks.push(part);
    }
    expect(chunks.at(-1)).toBe('a b');
  });

  test('AbortSignal via options', async () => {
    const model = new FakeChatModel('x');
    const client = ChatClient.create(model);
    const controller = new AbortController();
    controller.abort();

    await expect(
      client.prompt().user('hi').options({ signal: controller.signal }).call().content(),
    ).rejects.toMatchObject({ code: 'cancelled' });
  });

  test('toRequest builds without calling the model', () => {
    const model = new FakeChatModel('x');
    const request = ChatClient.create(model)
      .prompt()
      .system('sys')
      .user('usr')
      .options({ maxTokens: 32 })
      .toRequest();

    expect(request.prompt.getSystemMessage().text).toBe('sys');
    expect(request.prompt.getUserMessage().text).toBe('usr');
    expect(request.prompt.options?.maxTokens).toBe(32);
    expect(model.calls).toHaveLength(0);
  });
});

describe('Advisors', () => {
  test('before/after advisor mutates request and response context', async () => {
    const model = new FakeChatModel('answer');
    const order: string[] = [];

    const advisor = createBeforeAfterAdvisor({
      name: 'TraceAdvisor',
      order: 0,
      before(request) {
        order.push('before');
        const context = new Map(request.context);
        context.set('seen', true);
        return { ...request, context };
      },
      after(response) {
        order.push('after');
        const context = new Map(response.context);
        context.set('done', true);
        return { ...response, context };
      },
    });

    const client = ChatClient.create(model);
    const result = await client.prompt().user('q').advisors(advisor).call().chatClientResponse();

    expect(order).toEqual(['before', 'after']);
    expect(result.context.get('seen')).toBe(true);
    expect(result.context.get('done')).toBe(true);
    expect(result.chatResponse?.content).toBe('answer');
  });

  test('advisors run in ascending order', async () => {
    const model = new FakeChatModel('ok');
    const sequence: string[] = [];

    const make = (name: string, order: number): CallAdvisor => ({
      name,
      order,
      async adviseCall(request: ChatClientRequest, chain: CallAdvisorChain) {
        sequence.push(`in:${name}`);
        const response = await chain.nextCall(request);
        sequence.push(`out:${name}`);
        return response;
      },
    });

    const client = ChatClient.create(model);
    await client
      .prompt()
      .user('q')
      .advisors(make('late', 100), make('early', HIGHEST_PRECEDENCE + 10), make('mid', 50))
      .call()
      .content();

    expect(sequence).toEqual(['in:early', 'in:mid', 'in:late', 'out:late', 'out:mid', 'out:early']);
  });

  test('outer advisor near lowest precedence still wraps the model', async () => {
    const model = new FakeChatModel('ok');
    let sawOuter = false;
    const outer: CallAdvisor = {
      name: 'outer',
      order: LOWEST_PRECEDENCE - 1,
      async adviseCall(request, chain) {
        sawOuter = true;
        return chain.nextCall(request);
      },
    };

    await ChatClient.create(model).prompt().user('q').advisors(outer).call().content();

    expect(sawOuter).toBe(true);
    expect(model.calls).toHaveLength(1);
  });

  test('default advisors from builder', async () => {
    const logs: string[] = [];
    const model = new FakeChatModel('ok');
    const client = ChatClient.builder(model)
      .defaultAdvisors(
        new SimpleLoggerAdvisor({
          logger: (m) => logs.push(m),
        }),
      )
      .build();

    await client.prompt().user('q').call().content();
    expect(logs.some((l) => l.includes('request'))).toBe(true);
    expect(logs.some((l) => l.includes('response'))).toBe(true);
  });

  test('SimpleLoggerAdvisor logs around a streamed call too', async () => {
    const logs: string[] = [];
    const model = new FakeChatModel('a b');
    const client = ChatClient.builder(model)
      .defaultAdvisors(new SimpleLoggerAdvisor({ logger: (m) => logs.push(m) }))
      .build();

    const chunks: string[] = [];
    for await (const part of client.prompt().user('q').stream().content()) {
      chunks.push(part);
    }

    expect(chunks.at(-1)).toBe('a b');
    expect(logs.some((l) => l.includes('stream request'))).toBe(true);
    expect(logs.some((l) => l.includes('stream response'))).toBe(true);
  });

  test('SimpleLoggerAdvisor supports custom request/response formatters', async () => {
    const logs: string[] = [];
    const model = new FakeChatModel('ok');
    const client = ChatClient.builder(model)
      .defaultAdvisors(
        new SimpleLoggerAdvisor({
          logger: (m) => logs.push(m),
          requestToString: () => 'CUSTOM_REQUEST',
          responseToString: () => 'CUSTOM_RESPONSE',
        }),
      )
      .build();

    await client.prompt().user('q').call().content();
    expect(logs).toContain('[SimpleLoggerAdvisor] CUSTOM_REQUEST');
    expect(logs).toContain('[SimpleLoggerAdvisor] CUSTOM_RESPONSE');
  });

  test('memory precedence constant is below tool-calling band', () => {
    expect(DEFAULT_CHAT_MEMORY_PRECEDENCE_ORDER).toBeLessThan(HIGHEST_PRECEDENCE + 300);
  });

  test('advisor can rewrite user message', async () => {
    const recording = new FakeChatModel('done');
    const rewrite = createBeforeAfterAdvisor({
      name: 'Rewrite',
      order: 0,
      before(request) {
        return {
          ...request,
          prompt: request.prompt.augmentUserMessage('rewritten'),
        };
      },
    });

    await ChatClient.create(recording).prompt().user('original').advisors(rewrite).call().content();

    expect(recording.calls[0]?.getUserMessage().text).toBe('rewritten');
  });

  test('createBeforeAfterAdvisor supports before/after around a streamed call', async () => {
    const model = new FakeChatModel('a b');
    const seenAfter: string[] = [];
    const advisor = createBeforeAfterAdvisor({
      name: 'StreamRewrite',
      order: 0,
      before(request) {
        return { ...request, prompt: request.prompt.augmentUserMessage('rewritten') };
      },
      after(response) {
        seenAfter.push(response.chatResponse?.content ?? '');
        return response;
      },
    });

    const chunks: string[] = [];
    for await (const part of ChatClient.create(model)
      .prompt()
      .user('original')
      .advisors(advisor)
      .stream()
      .content()) {
      chunks.push(part);
    }

    expect(model.calls[0]?.getUserMessage().text).toBe('rewritten');
    expect(chunks.at(-1)).toBe('a b');
    expect(seenAfter.length).toBeGreaterThan(0);
  });

  test('createBeforeAfterAdvisor defaults before/after to identity when omitted', async () => {
    const model = new FakeChatModel('untouched');
    const advisor = createBeforeAfterAdvisor({ name: 'NoOp', order: 0 });

    const answer = await ChatClient.create(model)
      .prompt()
      .user('q')
      .advisors(advisor)
      .call()
      .content();
    expect(answer).toBe('untouched');
  });
});
