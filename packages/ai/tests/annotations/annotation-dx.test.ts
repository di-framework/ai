import { beforeEach, describe, expect, test } from 'bun:test';
import { useContainer } from '@di-framework/core/container';
import { Container as Injectable } from '@di-framework/core/decorators';
import {
  Agent,
  AiService,
  AiTokens,
  AssistantMessageAnn,
  ChatAgent,
  ChatClient,
  clearAnnotatedTypes,
  configureAi,
  FakeChatModel,
  MemoryId,
  MessageWindowChatMemory,
  OutputConverter,
  PromptTemplate,
  PromptVariable,
  processAiAnnotations,
  resolveAiService,
  resolveAnnotatedAgent,
  resolveChatClientBuilder,
  SchemaOutputConverter,
  ScriptedChatModel,
  StructuredOutput,
  SystemMessageAnn,
  Tool,
  ToolParam,
  ToolSet,
  toolCall,
  toolCallResponse,
  UserMessageAnn,
  V,
  WithMemory,
} from '../../src/index.ts';

beforeEach(() => {
  useContainer().clear();
  clearAnnotatedTypes();
});

describe('prototype ChatClient.Builder', () => {
  test('configureAi registers distinct builders per resolve', () => {
    configureAi({
      chatModel: new FakeChatModel('ok'),
      defaultSystem: 'base',
      scanAnnotations: false,
    });

    const a = resolveChatClientBuilder();
    const b = resolveChatClientBuilder();
    expect(a).not.toBe(b);

    const clientA = a.defaultSystem('A').build();
    const clientB = b.defaultSystem('B').build();
    expect(clientA).not.toBe(clientB);
  });
});

describe('@AiService', () => {
  test('proxy invokes ChatClient with system + user message', async () => {
    @AiService()
    class Friend {
      @SystemMessageAnn('You are a good friend. Answer using slang.')
      chat(@UserMessageAnn() _message: string): Promise<string> {
        throw new Error('AiService proxy should override this method');
      }
    }

    configureAi({
      chatModel: new FakeChatModel('yo whats up'),
      scanAnnotations: true,
    });

    const friend = resolveAiService(Friend);
    expect(await friend.chat('hello')).toBe('yo whats up');
  });

  test('wires @Tool beans declared on @AiService', async () => {
    @ToolSet()
    @Injectable()
    class LegalTools {
      @Tool({
        description: 'Returns last PRIVACY update',
        inputSchema: { type: 'object', properties: {} },
      })
      lastUpdatePrivacy(@ToolParam() _x?: string) {
        return '2013-03-09';
      }
    }

    @AiService({ tools: [LegalTools] })
    class CompanyBot {
      @SystemMessageAnn('You are a company policy bot.')
      ask(@UserMessageAnn() _question: string): Promise<string> {
        throw new Error('AiService proxy should override this method');
      }
    }

    const model = new ScriptedChatModel([
      {
        respond: (prompt) => {
          const tools = prompt.options?.toolCallbacks ?? [];
          expect(tools.some((t) => t.toolDefinition.name === 'lastUpdatePrivacy')).toBe(true);
          return toolCallResponse([toolCall('c1', 'lastUpdatePrivacy', {})]);
        },
      },
      { respond: 'Privacy was updated 2013-03-09' },
    ]);

    configureAi({
      chatModel: model,
      toolBeans: [LegalTools],
      scanAnnotations: true,
    });

    const bot = resolveAiService(CompanyBot);
    expect(await bot.ask('When was privacy updated?')).toBe('Privacy was updated 2013-03-09');
  });
});

describe('@Agent', () => {
  test('registers ChatAgent resolvable by class', async () => {
    @ToolSet()
    @Injectable()
    class WeatherTools {
      @Tool({
        description: 'Get weather',
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

    @Agent({
      system: 'You help with weather.',
      tools: [WeatherTools],
    })
    class SupportAgent {}

    const model = new ScriptedChatModel([
      {
        respond: () => toolCallResponse([toolCall('c1', 'getWeather', { city: 'Yorktown' })]),
      },
      { respond: '68F in Yorktown' },
    ]);

    configureAi({
      chatModel: model,
      toolBeans: [WeatherTools],
      scanAnnotations: true,
    });

    const agent = resolveAnnotatedAgent(SupportAgent);
    expect(agent).toBeInstanceOf(ChatAgent);
    const { content } = await agent.chat('Weather in Yorktown?');
    expect(content).toBe('68F in Yorktown');
  });

  test('ChatAgent.fromBuilder uses injected prototype builder', async () => {
    @Injectable()
    class WeatherTools {
      @Tool({
        description: 'Get weather',
        inputSchema: {
          type: 'object',
          properties: { city: { type: 'string' } },
          required: ['city'],
        },
      })
      getWeather({ city }: { city: string }) {
        return { temp: 72, city };
      }
    }

    const model = new ScriptedChatModel([
      {
        respond: () => toolCallResponse([toolCall('c1', 'getWeather', { city: 'Yorktown' })]),
      },
      { respond: '72F' },
    ]);

    // Tools are already on the prototype builder via toolBeans.
    configureAi({ chatModel: model, toolBeans: [WeatherTools], scanAnnotations: false });

    const agent = ChatAgent.fromBuilder(resolveChatClientBuilder())
      .system('Weather helper')
      .build();

    expect((await agent.chat('weather?')).content).toBe('72F');
  });

  test('createAgentFromAnnotations falls back gracefully when memory: true but nothing is registered', async () => {
    @Agent({ memory: true })
    class MemorylessAgent {}

    configureAi({
      chatModel: new FakeChatModel('no-memory-ok'),
      scanAnnotations: true,
    });

    const agent = resolveAnnotatedAgent(MemorylessAgent);
    expect((await agent.chat('hi')).content).toBe('no-memory-ok');
  });
});

describe('@AiService userText resolution branches', () => {
  test('renders a method-level @UserMessageAnn template (no param decorator) with @V args', async () => {
    @AiService()
    class UserTemplateService {
      @SystemMessageAnn('Be helpful.')
      @UserMessageAnn('Tell me about {topic}.')
      tellMe(@V('topic') _topic: string): Promise<string> {
        throw new Error('proxy override expected');
      }
    }
    const seenPrompts: string[] = [];
    const model = new ScriptedChatModel([
      {
        respond: (p) => {
          seenPrompts.push(p.getUserMessage().text ?? '');
          return 'told';
        },
      },
    ]);
    configureAi({ chatModel: model, scanAnnotations: true });
    const svc = resolveAiService(UserTemplateService);
    expect(await svc.tellMe('Yorktown')).toBe('told');
    expect(seenPrompts[0]).toBe('Tell me about Yorktown.');
  });

  test('falls back to a method-level @PromptTemplate rendered with @V/@PromptVariable args', async () => {
    @AiService()
    class TemplatedService {
      @SystemMessageAnn('Be helpful.')
      @PromptTemplate('Describe {city} in {mood} terms.')
      describe(@V('city') _city: string, @PromptVariable('mood') _mood: string): Promise<string> {
        throw new Error('proxy override expected');
      }
    }

    const seenPrompts: string[] = [];
    const model = new ScriptedChatModel([
      {
        respond: (p) => {
          seenPrompts.push(p.getUserMessage().text ?? '');
          return 'described';
        },
      },
    ]);

    configureAi({ chatModel: model, scanAnnotations: true });
    const svc = resolveAiService(TemplatedService);
    expect(await svc.describe('Yorktown', 'upbeat')).toBe('described');
    expect(seenPrompts[0]).toBe('Describe Yorktown in upbeat terms.');
  });

  test('falls back to a plain string argument when no user-message annotations are present', async () => {
    @AiService()
    class PlainStringService {
      ask(_q: string): Promise<string> {
        throw new Error('proxy override expected');
      }
    }
    const seenPrompts: string[] = [];
    const model = new ScriptedChatModel([
      {
        respond: (p) => {
          seenPrompts.push(p.getUserMessage().text ?? '');
          return 'plain-ok';
        },
      },
    ]);
    configureAi({ chatModel: model, scanAnnotations: true });
    const svc = resolveAiService(PlainStringService);
    expect(await svc.ask('bare string arg')).toBe('plain-ok');
    expect(seenPrompts[0]).toBe('bare string arg');
  });

  test('falls back to joining all args when there are no annotations and the first arg is not a string', async () => {
    @AiService()
    class JoinArgsService {
      compute(_a: number, _b: number): Promise<string> {
        throw new Error('proxy override expected');
      }
    }
    const seenPrompts: string[] = [];
    const model = new ScriptedChatModel([
      {
        respond: (p) => {
          seenPrompts.push(p.getUserMessage().text ?? '');
          return 'joined-ok';
        },
      },
    ]);
    configureAi({ chatModel: model, scanAnnotations: true });
    const svc = resolveAiService(JoinArgsService);
    expect(await svc.compute(1, 2)).toBe('joined-ok');
    expect(seenPrompts[0]).toBe('1 2');
  });

  test('assistant few-shot message is combined with a rendered system message', async () => {
    @AiService()
    class FewShotService {
      @SystemMessageAnn('Answer like a pirate.')
      @AssistantMessageAnn('Arr, that be a fine question.')
      ask(@UserMessageAnn() _q: string): Promise<string> {
        throw new Error('proxy override expected');
      }
    }
    const seenMessages: { messageType: string; text: string | null }[] = [];
    const model = new ScriptedChatModel([
      {
        respond: (p) => {
          seenMessages.push(
            ...p.messages.map((m) => ({ messageType: m.messageType, text: m.text })),
          );
          return 'ahoy';
        },
      },
    ]);
    configureAi({ chatModel: model, scanAnnotations: true });
    const svc = resolveAiService(FewShotService);
    expect(await svc.ask('where is the treasure?')).toBe('ahoy');
    expect(seenMessages.map((m) => m.messageType)).toEqual(['system', 'assistant', 'user']);
    expect(seenMessages[1]?.text).toBe('Arr, that be a fine question.');
  });
});

describe('@AiService structured output branches', () => {
  test('StructuredOutput schema with useProviderStructuredOutput/validateSchema configures the entity spec', async () => {
    @AiService()
    class StructuredService {
      @StructuredOutput({
        schema: { type: 'object', properties: { name: { type: 'string' } } },
        useProviderStructuredOutput: true,
        validateSchema: true,
      })
      getPerson(@UserMessageAnn() _q: string): Promise<{ name: string }> {
        throw new Error('proxy override expected');
      }
    }
    const model = new ScriptedChatModel([{ respond: '{"name":"Ada"}' }]);
    configureAi({ chatModel: model, scanAnnotations: true });
    const svc = resolveAiService(StructuredService);
    const person = await svc.getPerson('who?');
    expect(person).toEqual({ name: 'Ada' });
  });

  test('OutputConverter with a direct converter instance takes priority over StructuredOutput', async () => {
    const converter = new SchemaOutputConverter<{ ok: boolean }>({
      schema: { type: 'object', properties: { ok: { type: 'boolean' } } },
    });
    @AiService()
    class ConverterService {
      @OutputConverter(converter as never)
      @StructuredOutput({ schema: { type: 'object' } })
      check(@UserMessageAnn() _q: string): Promise<{ ok: boolean }> {
        throw new Error('proxy override expected');
      }
    }
    const model = new ScriptedChatModel([{ respond: '{"ok":true}' }]);
    configureAi({ chatModel: model, scanAnnotations: true });
    const svc = resolveAiService(ConverterService);
    expect(await svc.check('q')).toEqual({ ok: true });
  });

  test('a string-id OutputConverter with no StructuredOutput falls back to an empty schema object', async () => {
    @AiService()
    class StringConverterService {
      @OutputConverter('some-named-converter-id')
      run(@UserMessageAnn() _q: string): Promise<unknown> {
        throw new Error('proxy override expected');
      }
    }
    const model = new ScriptedChatModel([{ respond: '{}' }]);
    configureAi({ chatModel: model, scanAnnotations: true });
    const svc = resolveAiService(StringConverterService);
    // The schema falls back to `{}`, which accepts any JSON body.
    expect(await svc.run('q')).toEqual({});
  });
});

describe('registerFactoryForCtor container fallback', () => {
  test('falls back to registerOnContainer (and surfaces its error) when the container lacks registerFactory', () => {
    @AiService()
    class NoRegisterFactoryService {
      @SystemMessageAnn('hi')
      ask(@UserMessageAnn() _q: string): Promise<string> {
        throw new Error('proxy override expected');
      }
    }
    void NoRegisterFactoryService;

    const bareContainer = {
      resolve: () => {
        throw new Error('not used');
      },
    };

    expect(() => processAiAnnotations({ container: bareContainer as never })).toThrow(
      'Container does not support registerFactory',
    );
  });

  test('an explicit AiServiceOptions.chatClient token resolves that ChatClient directly from the container', async () => {
    @AiService({ chatClient: 'custom.chat.client' })
    class CustomClientService {
      ask(@UserMessageAnn() _q: string): Promise<string> {
        throw new Error('proxy override expected');
      }
    }

    const model = new FakeChatModel('via-custom-client');
    useContainer().registerFactory('custom.chat.client', () => ChatClient.create(model), {
      singleton: true,
    });
    configureAi({ chatModel: new FakeChatModel('unused-default'), scanAnnotations: true });

    const svc = resolveAiService(CustomClientService);
    expect(await svc.ask('hi')).toBe('via-custom-client');
  });
});

describe('@WithMemory + @MemoryId', () => {
  test('AiService passes conversation id to memory advisor', async () => {
    @AiService()
    @WithMemory()
    class RememberingBot {
      @SystemMessageAnn('Remember the user.')
      talk(@UserMessageAnn() _message: string, @MemoryId() _sessionId: string): Promise<string> {
        throw new Error('AiService proxy should override this method');
      }
    }

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
      scanAnnotations: true,
    });

    processAiAnnotations();

    const bot = resolveAiService(RememberingBot);
    await bot.talk("Hi, I'm Alice", 's1');
    expect(await bot.talk('What is my name?', 's1')).toBe('Your name is Alice');
  });
});

describe('AiTokens.CHAT_AGENT via configureAi', () => {
  test('registers default agent when agent: true', async () => {
    configureAi({
      chatModel: new FakeChatModel('agent-hi'),
      agent: true,
      scanAnnotations: false,
    });

    const agent = useContainer().resolve<ChatAgent>(AiTokens.CHAT_AGENT);
    expect((await agent.chat('hi')).content).toBe('agent-hi');
  });
});
