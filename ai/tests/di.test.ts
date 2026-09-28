import { beforeEach, describe, expect, test } from 'bun:test';
import { useContainer } from '@di-framework/core/container';
import { Component, Container as Injectable, Subscriber } from '@di-framework/core/decorators';
import {
  type AiChatResponseEvent,
  AiEvents,
  AiTokens,
  ChatAgent,
  ChatClient,
  type ChatModel,
  configureAi,
  type EmbeddingModel,
  EnableAi,
  enableAi,
  FakeChatModel,
  FakeEmbeddingModel,
  functionToolCallback,
  hasToolMethods,
  MessageWindowChatMemory,
  observationAdvisor,
  registerChatAgent,
  registerChatClient,
  registerChatModel,
  resolveChatAgent,
  resolveChatClient,
  resolveChatModel,
  ScriptedChatModel,
  SimpleVectorStore,
  Tool,
  toolCall,
  toolCallbackProviderFromBeans,
  toolCallbacksFromBean,
  toolCallbacksFromBeans,
  toolCallResponse,
  type VectorStore,
} from '../src/index.ts';

beforeEach(() => {
  useContainer().clear();
});

describe('registerChatModel / resolve', () => {
  test('registers model under string token and alias', () => {
    const model = new FakeChatModel('hello');
    registerChatModel(model, {
      aliases: [AiTokens.CHAT_MODEL_DEFAULT],
    });

    const c = useContainer();
    expect(c.resolve<ChatModel>(AiTokens.CHAT_MODEL)).toBe(model);
    expect(c.resolve<ChatModel>(AiTokens.CHAT_MODEL_DEFAULT)).toBe(model);
    expect(resolveChatModel().call).toBeTypeOf('function');
  });

  test('supports factory registration', async () => {
    registerChatModel(() => new FakeChatModel('from-factory'));
    const model = resolveChatModel();
    const client = ChatClient.create(model);
    expect(await client.prompt().user('x').call().content()).toBe('from-factory');
  });

  test('injects via @Component(token)', async () => {
    registerChatModel(new FakeChatModel('injected'));

    @Injectable()
    class Assistant {
      @Component(AiTokens.CHAT_MODEL)
      model!: ChatModel;

      async ask(q: string) {
        return ChatClient.create(this.model).prompt().user(q).call().content();
      }
    }

    const assistant = useContainer().resolve(Assistant);
    expect(await assistant.ask('hi')).toBe('injected');
  });
});

describe('configureAi auto-config', () => {
  test('registers ChatClient factory with default system', async () => {
    configureAi({
      chatModel: new FakeChatModel('ok'),
      defaultSystem: 'Be brief.',
    });

    const client = resolveChatClient();
    expect(client).toBeTruthy();
    expect(useContainer().resolve<ChatClient>(AiTokens.CHAT_CLIENT)).toBe(client);

    const content = await client.prompt().user('ping').call().content();
    expect(content).toBe('ok');
  });

  test('wires @Tool beans and tool-calling loop', async () => {
    @Injectable()
    class WeatherTools {
      @Tool({
        description: 'Get weather for a city',
        inputSchema: {
          type: 'object',
          properties: { city: { type: 'string' } },
          required: ['city'],
        },
      })
      getWeather({ city }: { city: string }) {
        return { temp: 68, city };
      }
    }

    const model = new ScriptedChatModel([
      {
        respond: (prompt) => {
          const tools = prompt.options?.toolCallbacks ?? [];
          expect(tools.some((t) => t.toolDefinition.name === 'getWeather')).toBe(true);
          return toolCallResponse([toolCall('c1', 'getWeather', { city: 'Yorktown' })]);
        },
      },
      { respond: '68F in Yorktown' },
    ]);

    configureAi({
      chatModel: model,
      toolBeans: [WeatherTools],
    });

    const tools = useContainer().resolve<unknown[]>(AiTokens.TOOL_CALLBACKS);
    expect(Array.isArray(tools)).toBe(true);
    expect(
      (tools as { toolDefinition: { name: string } }[]).map((t) => t.toolDefinition.name),
    ).toContain('getWeather');

    const answer = await resolveChatClient().prompt().user('Weather in Yorktown?').call().content();
    expect(answer).toBe('68F in Yorktown');
  });

  test('registers memory and ChatClient uses MessageChatMemoryAdvisor', async () => {
    const memory = MessageWindowChatMemory.builder().maxMessages(20).build();
    const model = new ScriptedChatModel([
      { respond: 'Hi Alice' },
      {
        respond: (p) => {
          const joined = p.messages.map((m) => m.text ?? '').join(' ');
          expect(joined).toContain('Alice');
          return 'Your name is Alice';
        },
      },
    ]);

    configureAi({
      chatModel: model,
      memory,
    });

    const client = resolveChatClient();
    const { CHAT_MEMORY_CONVERSATION_ID } = await import('../src/index.ts');
    await client
      .prompt()
      .user("Hi, I'm Alice")
      .advisorContext({ [CHAT_MEMORY_CONVERSATION_ID]: 'c1' })
      .call()
      .content();
    const second = await client
      .prompt()
      .user('What is my name?')
      .advisorContext({ [CHAT_MEMORY_CONVERSATION_ID]: 'c1' })
      .call()
      .content();
    expect(second).toBe('Your name is Alice');
  });
});

describe('@Tool decorator', () => {
  test('toolCallbacksFromBean binds methods', async () => {
    class LocalTools {
      @Tool({
        name: 'add',
        description: 'Add two numbers',
        inputSchema: {
          type: 'object',
          properties: {
            a: { type: 'number' },
            b: { type: 'number' },
          },
          required: ['a', 'b'],
        },
      })
      add({ a, b }: { a: number; b: number }) {
        return a + b;
      }
    }

    const callbacks = toolCallbacksFromBean(new LocalTools());
    expect(callbacks).toHaveLength(1);
    expect(callbacks[0]?.toolDefinition.name).toBe('add');
    const result = await callbacks[0]?.call(JSON.stringify({ a: 2, b: 3 }));
    expect(result).toBeDefined();
    expect(JSON.parse(result!)).toBe(5);
  });

  test('toolCallbacksFromBean throws for a null/non-object instance', () => {
    expect(() => toolCallbacksFromBean(null as unknown as object)).toThrow(
      'toolCallbacksFromBean requires a bean instance',
    );
    expect(() => toolCallbacksFromBean('noop' as unknown as object)).toThrow(
      'toolCallbacksFromBean requires a bean instance',
    );
  });

  test('toolCallbacksFromBean throws when the @Tool method is missing on the instance', () => {
    class BrokenTools {
      @Tool()
      missing() {
        return 'x';
      }
    }
    const instance = new BrokenTools();
    // Shadow the prototype method with `undefined` on the instance itself to
    // hit the "method missing" guard (deleting an inherited method is a no-op).
    (instance as unknown as Record<string, unknown>).missing = undefined;
    expect(() => toolCallbacksFromBean(instance)).toThrow(/is missing on/);
  });

  test('toolCallbacksFromBeans flattens tools across multiple bean instances', () => {
    class ToolsA {
      @Tool({ name: 'a' })
      a() {
        return 'a';
      }
    }
    class ToolsB {
      @Tool({ name: 'b' })
      b() {
        return 'b';
      }
    }
    const callbacks = toolCallbacksFromBeans(new ToolsA(), new ToolsB());
    expect(callbacks.map((c) => c.toolDefinition.name)).toEqual(['a', 'b']);
  });

  test('toolCallbackProviderFromBeans builds a provider over multiple bean instances', async () => {
    class ToolsC {
      @Tool({ name: 'c' })
      c() {
        return 'c-result';
      }
    }
    const provider = toolCallbackProviderFromBeans(new ToolsC());
    const callbacks = provider.getToolCallbacks();
    expect(callbacks).toHaveLength(1);
    expect(await callbacks[0]?.call('{}')).toContain('c-result');
  });

  test('hasToolMethods reflects presence/absence of @Tool methods', () => {
    class WithTool {
      @Tool()
      go() {
        return 'go';
      }
    }
    class WithoutTool {}
    expect(hasToolMethods(WithTool)).toBe(true);
    expect(hasToolMethods(WithoutTool)).toBe(false);
  });
});

describe('ObservationAdvisor + @Subscriber', () => {
  test('emits redacted chat events without prompt text by default', async () => {
    const events: unknown[] = [];

    @Injectable()
    class AiAudit {
      @Subscriber(AiEvents.CHAT_REQUEST)
      onRequest(payload: unknown) {
        events.push(payload);
      }

      @Subscriber(AiEvents.CHAT_RESPONSE)
      onResponse(payload: unknown) {
        events.push(payload);
      }
    }

    useContainer().resolve(AiAudit);

    const model = new FakeChatModel('pong');
    const client = ChatClient.builder(model).defaultAdvisors(observationAdvisor()).build();

    await client.prompt().user('secret prompt text').call().content();

    expect(events.length).toBe(2);
    const req = events[0] as Record<string, unknown>;
    const res = events[1] as AiChatResponseEvent;
    expect(req.type).toBe(AiEvents.CHAT_REQUEST);
    expect(req.messageCount).toBe(1);
    expect(req.promptText).toBeUndefined();
    expect(res.type).toBe(AiEvents.CHAT_RESPONSE);
    expect(res.responseText).toBeUndefined();
    expect(res.durationMs).toBeGreaterThanOrEqual(0);
  });

  test('configureAi({ observation: true }) wires observer', async () => {
    const seen: string[] = [];

    @Injectable()
    class Sink {
      @Subscriber(AiEvents.CHAT_RESPONSE)
      onRes(payload: AiChatResponseEvent) {
        seen.push(payload.type);
      }
    }

    useContainer().resolve(Sink);

    configureAi({
      chatModel: new FakeChatModel('x'),
      observation: true,
    });

    await resolveChatClient().prompt().user('q').call().content();
    expect(seen).toEqual([AiEvents.CHAT_RESPONSE]);
  });

  test('includePromptText/includeResponseText attach truncated text', async () => {
    const events: unknown[] = [];

    @Injectable()
    class TextAudit {
      @Subscriber(AiEvents.CHAT_REQUEST)
      onRequest(payload: unknown) {
        events.push(payload);
      }

      @Subscriber(AiEvents.CHAT_RESPONSE)
      onResponse(payload: unknown) {
        events.push(payload);
      }
    }

    useContainer().resolve(TextAudit);

    const longText = 'x'.repeat(20);
    const model = new FakeChatModel(longText);
    const client = ChatClient.builder(model)
      .defaultAdvisors(
        observationAdvisor({
          includePromptText: true,
          includeResponseText: true,
          maxTextLength: 5,
        }),
      )
      .build();

    await client.prompt().user('a long prompt text here').call().content();

    const req = events[0] as { promptText?: string };
    const res = events[1] as AiChatResponseEvent;
    expect(req.promptText?.length).toBe(6); // 5 chars + ellipsis
    expect(req.promptText?.endsWith('\u2026')).toBe(true);
    expect(res.responseText?.length).toBe(6);
    expect(res.responseText?.endsWith('\u2026')).toBe(true);
  });

  test('does not truncate text shorter than maxTextLength', async () => {
    const events: unknown[] = [];

    @Injectable()
    class ShortTextAudit {
      @Subscriber(AiEvents.CHAT_REQUEST)
      onRequest(payload: unknown) {
        events.push(payload);
      }
    }
    useContainer().resolve(ShortTextAudit);

    const model = new FakeChatModel('hi');
    const client = ChatClient.builder(model)
      .defaultAdvisors(observationAdvisor({ includePromptText: true, maxTextLength: 500 }))
      .build();

    await client.prompt().user('short').call().content();
    const req = events[0] as { promptText?: string };
    expect(req.promptText).toBe('short');
  });

  test('emits AiEvents.CHAT_ERROR and rethrows when the chain fails', async () => {
    const events: unknown[] = [];

    @Injectable()
    class ErrorAudit {
      @Subscriber(AiEvents.CHAT_ERROR)
      onError(payload: unknown) {
        events.push(payload);
      }
    }
    useContainer().resolve(ErrorAudit);

    const model = new FakeChatModel(() => {
      throw new Error('boom');
    });
    const client = ChatClient.builder(model).defaultAdvisors(observationAdvisor()).build();

    await expect(client.prompt().user('q').call().content()).rejects.toThrow('boom');
    expect(events).toHaveLength(1);
    const err = events[0] as { type: string; errorMessage: string; errorName: string };
    expect(err.type).toBe(AiEvents.CHAT_ERROR);
    expect(err.errorMessage).toBe('boom');
    expect(err.errorName).toBe('Error');
  });

  test('emits CHAT_ERROR for non-Error thrown values', async () => {
    const events: unknown[] = [];

    @Injectable()
    class ErrorAudit2 {
      @Subscriber(AiEvents.CHAT_ERROR)
      onError(payload: unknown) {
        events.push(payload);
      }
    }
    useContainer().resolve(ErrorAudit2);

    const model = new FakeChatModel(() => {
      // eslint-disable-next-line @typescript-eslint/no-throw-literal
      throw 'plain string failure';
    });
    const client = ChatClient.builder(model).defaultAdvisors(observationAdvisor()).build();

    await expect(client.prompt().user('q').call().content()).rejects.toBeDefined();
    expect(events).toHaveLength(1);
    const err = events[0] as { errorMessage: string };
    expect(err.errorMessage).toBe('plain string failure');
  });

  test('adviseStream emits request/response events around a streamed call', async () => {
    const events: unknown[] = [];

    @Injectable()
    class StreamAudit {
      @Subscriber(AiEvents.CHAT_REQUEST)
      onRequest(payload: unknown) {
        events.push(payload);
      }

      @Subscriber(AiEvents.CHAT_RESPONSE)
      onResponse(payload: unknown) {
        events.push(payload);
      }
    }
    useContainer().resolve(StreamAudit);

    const model = new FakeChatModel('a b c');
    const client = ChatClient.builder(model).defaultAdvisors(observationAdvisor()).build();

    const chunks: string[] = [];
    for await (const part of client.prompt().user('q').stream().content()) {
      chunks.push(part);
    }

    expect(chunks.at(-1)).toBe('a b c');
    expect(events).toHaveLength(2);
    expect((events[0] as { type: string }).type).toBe(AiEvents.CHAT_REQUEST);
    expect((events[1] as { type: string }).type).toBe(AiEvents.CHAT_RESPONSE);
  });

  test('emitRequest reports tool names when tools are present', async () => {
    const events: unknown[] = [];

    @Injectable()
    class ToolAudit {
      @Subscriber(AiEvents.CHAT_REQUEST)
      onRequest(payload: unknown) {
        events.push(payload);
      }
    }
    useContainer().resolve(ToolAudit);

    const model = new FakeChatModel('ok');
    const client = ChatClient.builder(model).defaultAdvisors(observationAdvisor()).build();
    const tool = functionToolCallback({ name: 'noop', call: () => 'ok' });

    await client.prompt().user('q').tools(tool).call().content();

    const req = events[0] as { hasTools: boolean; toolNames: string[] };
    expect(req.hasTools).toBe(true);
    expect(req.toolNames).toEqual(['noop']);
  });

  test('observationAdvisor accepts an explicit container and tolerates missing emit()', () => {
    const noEmitContainer = {} as never;
    const advisor = observationAdvisor({ container: noEmitContainer });
    expect(advisor.name).toBe('AI Observation Advisor');
  });
});

describe('registerChatClient manual', () => {
  test('registers prebuilt client', async () => {
    const model = new FakeChatModel('manual');
    const client = ChatClient.create(model);
    registerChatClient(client);
    expect(await resolveChatClient().prompt().user('a').call().content()).toBe('manual');
  });

  test('explicit tools still work alongside DI client', async () => {
    const weather = functionToolCallback({
      name: 'getWeather',
      call: () => ({ temp: 1 }),
    });
    configureAi({
      chatModel: new FakeChatModel('no-tools-path'),
      tools: [weather],
    });
    const registered = useContainer().resolve<{ toolDefinition: { name: string } }[]>(
      AiTokens.TOOL_CALLBACKS,
    );
    expect(registered[0]?.toolDefinition.name).toBe('getWeather');
  });
});

describe('resolveChatAgent / registerChatAgent', () => {
  test('resolveChatAgent resolves the agent registered by configureAi({ agent: true })', async () => {
    configureAi({
      chatModel: new FakeChatModel('agent-reply'),
      agent: true,
      scanAnnotations: false,
    });
    const agent = resolveChatAgent();
    expect((await agent.chat('hi')).content).toBe('agent-reply');
  });

  test('registerChatAgent supports a custom token/aliases and a direct (non-factory) agent', async () => {
    const model = new FakeChatModel('direct-agent');
    const agent = ChatAgent.create({ chatModel: model });
    registerChatAgent(agent, { token: 'custom.agent', aliases: ['custom.agent.alias'] });
    expect(useContainer().resolve<ChatAgent>('custom.agent')).toBe(agent);
    expect(useContainer().resolve<ChatAgent>('custom.agent.alias')).toBe(agent);
  });
});

describe('configureAi embeddingModel / vectorStore registration', () => {
  test('registers an embeddingModel and vectorStore under their AiTokens', () => {
    const embeddingModel = new FakeEmbeddingModel();
    const vectorStore = SimpleVectorStore.of(embeddingModel);
    configureAi({
      chatModel: new FakeChatModel('x'),
      embeddingModel,
      vectorStore,
      scanAnnotations: false,
    });
    expect(useContainer().resolve<EmbeddingModel>(AiTokens.EMBEDDING_MODEL)).toBe(embeddingModel);
    expect(useContainer().resolve<VectorStore>(AiTokens.VECTOR_STORE)).toBe(vectorStore);
  });
});

describe('configureAi defaultOptions', () => {
  test('applies defaultOptions to the built ChatClient', async () => {
    let seenTemperature: number | undefined;
    const model = new ScriptedChatModel([
      {
        respond: (p) => {
          seenTemperature = p.options?.temperature;
          return 'ok';
        },
      },
    ]);
    configureAi({
      chatModel: model,
      defaultOptions: { temperature: 0.42 },
      scanAnnotations: false,
    });
    await resolveChatClient().prompt().user('hi').call().content();
    expect(seenTemperature).toBe(0.42);
  });
});

describe('enableAi', () => {
  test('merges @EnableAi options from the app class with explicit overrides', async () => {
    @EnableAi({ chatModel: new FakeChatModel('from-annotation') })
    class App {}

    const result = enableAi(App, { scanAnnotations: false });
    const client = result.chatClientToken
      ? useContainer().resolve<ChatClient>(result.chatClientToken)
      : undefined;
    expect(await client?.prompt().user('q').call().content()).toBe('from-annotation');
  });
});
