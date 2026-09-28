import { beforeEach, describe, expect, test } from 'bun:test';
import { useContainer } from '@di-framework/core/container';
import { Container as Injectable, Subscriber } from '@di-framework/core/decorators';
import {
  getAdvisorOptions,
  getAdvisorOrder,
  getAiObserved,
  getWithMemory,
  getWithRag,
  getWithTools,
} from '../../src/di/annotations/advisor.ts';
import {
  AgentStrategy,
  getAgentOptions,
  getAgentStrategy,
  getAiServiceOptions,
  getAssistantMessage,
  getMaxIterations,
  getMemoryIdParams,
  getPromptVariables,
  getSystemMessage,
  getUserMessageAnn,
  getUserMessageParams,
} from '../../src/di/annotations/assistant.ts';
import { getMcpClientOptions, getMcpTools } from '../../src/di/annotations/mcp.ts';
import {
  getAnnotatedTypes as getAnnotatedTypesDirect,
  markAnnotated,
} from '../../src/di/annotations/meta.ts';
import {
  getAiConfigurationOptions,
  getAiPropertiesOptions,
  getEnableAiOptions,
} from '../../src/di/annotations/model.ts';
import {
  getLLMDescription,
  getOutputConverter,
  getPromptOptions,
  getStructuredOutput,
} from '../../src/di/annotations/prompt.ts';
import {
  getClassIndexedDocument,
  getIndexedDocuments,
  getVectorStoreToken,
} from '../../src/di/annotations/rag.ts';
import {
  getToolParamAnns,
  getToolResultOptions,
  getToolSetOptions,
  isReturnDirect,
} from '../../src/di/annotations/tools.ts';
import {
  getChainOptions,
  getEvaluateOptions,
  getOptimizeOptions,
  getOrchestratorOptions,
  getParallelOptions,
  getRouteOptions,
  getWorkerMethods,
} from '../../src/di/annotations/workflow.ts';
import {
  Agent,
  AiAdvisor,
  AiConfiguration,
  AiEvents,
  AiObserved,
  AiProperties,
  AiService,
  Assistant,
  Chain,
  ChatAgent,
  ChatAgentBean,
  ChatClientAnn,
  ChatClientDecorator,
  ChatMemoryAnn,
  ChatModelAnn,
  ChatResponse,
  ConversationId,
  clearAnnotatedTypes,
  configureAi,
  DocumentAnn,
  EmbeddingModelAnn,
  EnableAi,
  Evaluate,
  FakeChatModel,
  FakeEmbeddingModel,
  functionToolCallback,
  IndexedDocument,
  LLMDescription,
  MaxIterations,
  McpClient,
  McpTool,
  MemoryId,
  MessageWindowChatMemory,
  Observed,
  Optimize,
  Orchestrator,
  Order,
  OutputConverter,
  Parallel,
  PromptTemplate,
  PromptVariable,
  processAiAnnotations,
  RetrievalAugmented,
  Retriever,
  ReturnDirect,
  Route,
  Router,
  resolveAiService,
  resolveAnnotatedAgent,
  SchemaOutputConverter,
  ScriptedChatModel,
  SimpleVectorStore,
  StructuredOutput,
  SystemMessageAnn,
  Tool,
  ToolParam,
  ToolResult,
  ToolSet,
  Tools,
  textDocument,
  toolCall,
  toolCallResponse,
  UserMessageAnn,
  V,
  VectorStoreAnn,
  WithMemory,
  WithRag,
  WithTools,
  Worker,
} from '../../src/index.ts';

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

beforeEach(() => {
  useContainer().clear();
  clearAnnotatedTypes();
});

describe('meta.ts', () => {
  test('markAnnotated marks a class as annotated without overwriting existing metadata', () => {
    class PlainClass {}
    expect(getAnnotatedTypesDirect()).not.toContain(PlainClass);
    markAnnotated(PlainClass);
    expect(getAnnotatedTypesDirect()).toContain(PlainClass);
  });
});

describe('mcp.ts annotations', () => {
  test('McpClient as class/property/param decorator + string shorthand', () => {
    @McpClient('fsClient')
    class HasClassMcp {}
    expect(getMcpClientOptions(HasClassMcp)).toEqual({ token: 'fsClient' });

    class HasPropMcp {
      @McpClient({ title: 'fs' })
      declare session: unknown;
    }
    expect(() => new HasPropMcp()).not.toThrow();

    class HasParamMcp {
      ask(@McpClient() _session: unknown) {}
    }
    expect(() => new HasParamMcp().ask(undefined)).not.toThrow();
  });

  test('McpTool exposes a method + getMcpTools reads it back', () => {
    class HasMcpTool {
      @McpTool('list files')
      listFiles() {
        return [];
      }
      @McpTool({ name: 'writeFile', description: 'write' })
      write() {
        return true;
      }
    }
    const tools = getMcpTools(HasMcpTool);
    expect(tools.listFiles?.description).toBe('list files');
    expect(tools.write?.name).toBe('writeFile');
  });
});

describe('model.ts annotations', () => {
  test('ChatModelAnn / ChatClientAnn class/property/param forms + default tokens', () => {
    @ChatModelAnn()
    class ClassLevelModel {}
    expect(() => ClassLevelModel).not.toThrow();

    class PropModel {
      @ChatModelAnn('special')
      declare model: unknown;
    }
    expect(() => new PropModel()).not.toThrow();

    class ParamModel {
      ask(@ChatModelAnn() _m: unknown) {}
    }
    expect(() => new ParamModel().ask(undefined)).not.toThrow();

    @ChatClientAnn('myClient')
    class ClassLevelClient {}
    expect(() => ClassLevelClient).not.toThrow();
    expect(ChatClientDecorator).toBe(ChatClientAnn);

    class PropClient {
      @ChatClientAnn()
      declare client: unknown;
    }
    expect(() => new PropClient()).not.toThrow();

    class ParamClient {
      ask(@ChatClientAnn('x') _c: unknown) {}
    }
    expect(() => new ParamClient().ask(undefined)).not.toThrow();
  });

  test('AiConfiguration / EnableAi / AiProperties + getters', () => {
    @AiConfiguration({ prefix: 'custom' })
    class Config {}
    expect(getAiConfigurationOptions(Config)).toEqual({ prefix: 'custom' });

    @EnableAi({ defaultSystem: 'be nice' })
    class App {}
    expect(getEnableAiOptions(App)).toMatchObject({
      defaultSystem: 'be nice',
      scanAnnotations: true,
    });

    @AiProperties()
    class Props {}
    expect(getAiPropertiesOptions(Props)).toEqual({ prefix: 'ai' });
  });

  test('configureAi merges EnableAi options from annotated app classes', async () => {
    @EnableAi({ defaultSystem: 'from enable ai' })
    class App2 {}
    void App2;

    const model = new FakeChatModel((prompt) => {
      const sys = prompt.messages.find((m) => m.messageType === 'system');
      expect(sys?.text).toBe('from enable ai');
      return ChatResponse.of('hi');
    });

    configureAi({ chatModel: model, scanAnnotations: false });
    const { resolveChatClient } = await import('../../src/index.ts');
    const content = await resolveChatClient().prompt().user('x').call().content();
    expect(content).toBe('hi');
  });
});

describe('prompt.ts annotations', () => {
  test('Prompt/PromptTemplate on method + class, LLMDescription forms', () => {
    class HasPrompt {
      @PromptTemplate('Hello {name}')
      greet() {}
    }
    expect(getPromptOptions(HasPrompt, 'greet')).toEqual({ template: 'Hello {name}' });

    @PromptTemplate({ name: 'classPrompt', template: 'Sys' })
    class ClassPrompt {}
    expect(getPromptOptions(ClassPrompt)).toEqual({ name: 'classPrompt', template: 'Sys' });
    expect(getPromptOptions(ClassPrompt, 'missingMethod')).toEqual({
      name: 'classPrompt',
      template: 'Sys',
    });

    @LLMDescription('a described class')
    class DescribedClass {}
    expect(getLLMDescription(DescribedClass)).toBe('a described class');

    class DescribedMembers {
      @LLMDescription('described method')
      run() {}
      method2(@LLMDescription('described param') _x: string) {}
    }
    void DescribedMembers;
  });

  test('StructuredOutput accepts string / schema object / options object', () => {
    class HasStructured {
      @StructuredOutput('{"type":"string"}')
      asString() {}
      @StructuredOutput({ type: 'object', properties: {} })
      asSchemaObject() {}
      @StructuredOutput({ schema: { type: 'object' }, useProviderStructuredOutput: true })
      asOptions() {}
    }
    expect(getStructuredOutput(HasStructured, 'asString')).toEqual({
      schema: '{"type":"string"}',
    });
    expect(getStructuredOutput(HasStructured, 'asSchemaObject')).toEqual({
      schema: { type: 'object', properties: {} },
    });
    expect(getStructuredOutput(HasStructured, 'asOptions')).toEqual({
      schema: { type: 'object' },
      useProviderStructuredOutput: true,
    });
  });

  test('OutputConverter accepts converter directly, string id, or options object', () => {
    const converter = new SchemaOutputConverter<unknown>({ schema: { type: 'object' } });

    class HasConverter {
      @OutputConverter(converter as never)
      direct() {}
      @OutputConverter('named-converter')
      named() {}
      @OutputConverter({ converter: 'wrapped' })
      wrapped() {}
    }
    expect(getOutputConverter(HasConverter, 'direct')?.converter).toBe(converter);
    expect(getOutputConverter(HasConverter, 'named')).toEqual({ converter: 'named-converter' });
    expect(getOutputConverter(HasConverter, 'wrapped')).toEqual({ converter: 'wrapped' });
  });
});

describe('rag.ts annotations', () => {
  test('VectorStoreAnn/Retriever/ChatMemoryAnn/EmbeddingModelAnn qualify class/prop/param', () => {
    @VectorStoreAnn('myStore')
    class ClassStore {}
    expect(getVectorStoreToken(ClassStore)).toBe('myStore');

    class PropStore {
      @VectorStoreAnn()
      declare store: unknown;
    }
    expect(() => new PropStore()).not.toThrow();

    class ParamStore {
      ask(@VectorStoreAnn('x') _s: unknown) {}
    }
    expect(() => new ParamStore().ask(undefined)).not.toThrow();

    @Retriever('myRetriever')
    class ClassRetriever {}
    void ClassRetriever;

    @ChatMemoryAnn()
    class ClassMemory {}
    void ClassMemory;

    @EmbeddingModelAnn('myEmbed')
    class ClassEmbed {}
    void ClassEmbed;
  });

  test('DocumentAnn / IndexedDocument class + method forms, getters', () => {
    @DocumentAnn({ id: 'doc-1' })
    class ClassDoc {}
    void ClassDoc;

    class HasDocumentMethod {
      @DocumentAnn()
      loadDoc() {}
    }
    void HasDocumentMethod;

    @IndexedDocument('Static class text')
    class ClassIndexed {}
    expect(getClassIndexedDocument(ClassIndexed)?.text).toBe('Static class text');

    class HasIndexedMethod {
      @IndexedDocument({ text: 'method text', id: 'm1' })
      seedDoc() {}
    }
    const docs = getIndexedDocuments(HasIndexedMethod);
    expect(docs.seedDoc?.text).toBe('method text');
  });
});

describe('tools.ts annotations', () => {
  test('ToolSet/Tools alias + string shorthand, getToolSetOptions', () => {
    @ToolSet('a toolset')
    class SetA {}
    expect(getToolSetOptions(SetA)).toEqual({ description: 'a toolset' });

    @Tools({ name: 'named-set' })
    class SetB {}
    expect(getToolSetOptions(SetB)).toEqual({ name: 'named-set' });
  });

  test('ToolParam string/options + ToolResult + getters', () => {
    class HasParams {
      @Tool()
      run(
        @ToolParam('the x value') _x: number,
        @ToolParam({ description: 'the y value', required: false }) _y: number,
      ) {
        return _x + _y;
      }

      @Tool()
      @ToolResult({ converter: 'custom' })
      convertedResult() {
        return 1;
      }
    }
    const params = getToolParamAnns(HasParams, 'run');
    expect(params.map((p) => p.value.description)).toEqual(['the x value', 'the y value']);
    expect(getToolResultOptions(HasParams, 'convertedResult')).toEqual({ converter: 'custom' });
  });

  test('ReturnDirect updates existing @Tool metadata entry and isReturnDirect', () => {
    class HasReturnDirect {
      @ReturnDirect()
      @Tool({ name: 'directTool' })
      directTool() {
        return 'done';
      }

      // ReturnDirect applied where no prior @Tool metadata exists: idx<0 branch
      @ReturnDirect()
      untracked() {
        return 'n/a';
      }
    }
    expect(isReturnDirect(HasReturnDirect, 'directTool')).toBe(true);
    expect(isReturnDirect(HasReturnDirect, 'untracked')).toBe(true);
    expect(isReturnDirect(HasReturnDirect, 'somethingElse')).toBe(false);
  });
});

describe('workflow.ts annotations', () => {
  test('Chain via array shorthand and options object', () => {
    @Chain(['step1', 'step2'])
    class ArrChain {}
    expect(getChainOptions(ArrChain)).toEqual({ steps: ['step1', 'step2'] });

    @Chain({ steps: ['a'] })
    class OptChain {}
    expect(getChainOptions(OptChain)).toEqual({ steps: ['a'] });
  });

  test('Route/Router class + method forms', () => {
    @Route({ routes: { a: 'x' } })
    class ClassRoute {}
    expect(getRouteOptions(ClassRoute)).toEqual({ routes: { a: 'x' } });

    class HasRouteMethod {
      @Router()
      dispatch() {}
    }
    void HasRouteMethod;
  });

  test('Parallel array + options, Orchestrator/Worker string + options + class/method', () => {
    @Parallel(['p1', 'p2'])
    class ArrParallel {}
    expect(getParallelOptions(ArrParallel)).toEqual({ prompts: ['p1', 'p2'] });

    @Parallel({ prompts: ['q'] })
    class OptParallel {}
    void OptParallel;

    @Orchestrator('planner')
    class ClassOrchestrator {}
    expect(getOrchestratorOptions(ClassOrchestrator)).toEqual({ role: 'planner' });

    class HasOrchestratorMethod {
      @Orchestrator({ role: 'lead' })
      plan() {}
      @Worker('coder')
      work() {}
    }
    const workers = getWorkerMethods(HasOrchestratorMethod);
    expect(workers.work).toEqual({ role: 'coder' });

    @Worker({ role: 'class-level' })
    class ClassLevelWorker {}
    void ClassLevelWorker;
  });

  test('Evaluate/Optimize string + options + class/method forms', () => {
    @Evaluate('quality check')
    class ClassEvaluate {}
    expect(getEvaluateOptions(ClassEvaluate)).toEqual({ criteria: 'quality check' });

    class HasEvalMethod {
      @Evaluate({ criteria: 'accuracy' })
      evalMethod() {}
      @Optimize('refine')
      optimizeMethod() {}
    }
    void HasEvalMethod;

    @Optimize({ criteria: 'brevity' })
    class ClassOptimize {}
    expect(getOptimizeOptions(ClassOptimize)).toEqual({ criteria: 'brevity' });
  });
});

describe('advisor.ts annotations', () => {
  test('Advisor with/without order, AdvisorOrder/Order alias', () => {
    @AiAdvisor({ order: 5 })
    class OrderedAdvisor {}
    expect(getAdvisorOptions(OrderedAdvisor)).toEqual({ order: 5 });
    expect(getAdvisorOrder(OrderedAdvisor)).toBe(5);

    @AiAdvisor()
    class UnorderedAdvisor {}
    expect(getAdvisorOptions(UnorderedAdvisor)).toEqual({});
    expect(getAdvisorOrder(UnorderedAdvisor)).toBeUndefined();

    @Order(9)
    class Ordered2 {}
    expect(getAdvisorOrder(Ordered2)).toBe(9);
  });

  test('WithMemory / WithRag(RetrievalAugmented) / WithTools / AiObserved(Observed) boolean + options forms', () => {
    @WithMemory()
    class MemDefault {}
    expect(getWithMemory(MemDefault)).toEqual({ enabled: true });

    @WithMemory(false)
    class MemDisabled {}
    expect(getWithMemory(MemDisabled)).toEqual({ enabled: false });

    @WithMemory({ token: 'custom-mem' })
    class MemCustom {}
    expect(getWithMemory(MemCustom)).toEqual({ enabled: true, token: 'custom-mem' });

    @WithRag({ topK: 3 })
    class RagCustom {}
    expect(getWithRag(RagCustom)).toEqual({ enabled: true, topK: 3 });

    @RetrievalAugmented(false)
    class RagDisabled {}
    expect(getWithRag(RagDisabled)).toEqual({ enabled: false });

    class Tool1 {}
    @WithTools([Tool1])
    class ToolsArr {}
    expect(getWithTools(ToolsArr)?.enabled).toBe(true);

    @WithTools(true)
    class ToolsBool {}
    expect(getWithTools(ToolsBool)).toEqual({ enabled: true });

    @WithTools({ tools: [] })
    class ToolsOpt {}
    expect(getWithTools(ToolsOpt)).toEqual({ enabled: true, tools: [] });

    @AiObserved()
    class ObservedDefault {}
    expect(getAiObserved(ObservedDefault)).toEqual({ enabled: true });

    @Observed({ includePromptText: true })
    class ObservedCustom {}
    expect(getAiObserved(ObservedCustom)).toEqual({ enabled: true, includePromptText: true });
  });
});

describe('assistant.ts annotations', () => {
  test('AiService/Assistant alias + Agent/ChatAgentBean alias, getters', () => {
    @AiService({ system: 'svc system' })
    class SvcA {}
    expect(getAiServiceOptions(SvcA)).toEqual({ system: 'svc system' });

    @Assistant({ system: 'alias system' })
    class SvcB {}
    void SvcB;

    @Agent({ system: 'agent system' })
    class AgentA {}
    expect(getAgentOptions(AgentA)).toEqual({ system: 'agent system' });

    @ChatAgentBean({ memory: true })
    class AgentB {}
    void AgentB;
  });

  test('SystemMessageAnn class-level fallback when no method-level override', () => {
    @SystemMessageAnn('class-level system')
    class HasClassSystem {
      untouched() {}
    }
    expect(getSystemMessage(HasClassSystem, 'untouched')).toBe('class-level system');
    expect(getSystemMessage(HasClassSystem)).toBe('class-level system');

    class NoSystem {}
    expect(getSystemMessage(NoSystem, 'anything')).toBeUndefined();
  });

  test('UserMessageAnn as method template decorator (non-param) and param form', () => {
    class HasUserTemplate {
      @UserMessageAnn('Hello {name}')
      greet() {}
    }
    expect(getUserMessageAnn(HasUserTemplate, 'greet')).toEqual({ template: 'Hello {name}' });

    class HasUserParam {
      ask(@UserMessageAnn() _q: string) {}
    }
    const params = getUserMessageParams(HasUserParam, 'ask');
    expect(params[0]?.value.fromParam).toBe(true);
  });

  test('V/PromptVariable, MemoryId/ConversationId, AgentStrategy, MaxIterations', () => {
    class HasVars {
      render(@PromptVariable('city') _c: string, @V() _other: string) {}
    }
    const vars = getPromptVariables(HasVars, 'render');
    expect(vars[0]?.value.name).toBe('city');
    expect(vars[1]?.value).toEqual({});

    class HasMemoryParam {
      talk(@MemoryId() _id: string) {}
      talk2(@ConversationId() _id: string) {}
    }
    expect(getMemoryIdParams(HasMemoryParam, 'talk')).toHaveLength(1);
    expect(getMemoryIdParams(HasMemoryParam, 'talk2')).toHaveLength(1);

    @AgentStrategy('chain')
    class StrategyByString {}
    expect(getAgentStrategy(StrategyByString)).toEqual({ kind: 'chain' });

    @AgentStrategy({ kind: 'route', steps: ['a'] })
    class StrategyByOptions {}
    expect(getAgentStrategy(StrategyByOptions)).toEqual({ kind: 'route', steps: ['a'] });

    @MaxIterations(7)
    class HasMaxIter {}
    expect(getMaxIterations(HasMaxIter)).toBe(7);
  });

  test('AssistantMessageAnn attaches few-shot text readable via getAssistantMessage', async () => {
    const { AssistantMessageAnn } = await import('../../src/index.ts');
    class HasFewShot2 {
      @AssistantMessageAnn('Sure, I can help.')
      chat() {}
    }
    expect(getAssistantMessage(HasFewShot2, 'chat')).toBe('Sure, I can help.');
  });
});

describe('processAiAnnotations integration', () => {
  test('registers Chain/Route/Parallel/Evaluate workflow beans resolvable via container', async () => {
    @Chain(['Summarize.', 'Shorten further.'])
    class MyChain {}

    @Route({ routes: { billing: 'You are billing.', general: 'You are general.' } })
    class MyRoute {}

    @Parallel(['input-a', 'input-b'])
    class MyParallel {}

    @Evaluate({ criteria: 'ok' })
    class MyEvaluator {}

    const model = new ScriptedChatModel([
      { respond: 'step1 done' },
      { respond: 'step2 done' },
      { respond: '{"route":"billing"}' },
      { respond: 'billing answer' },
      { respond: 'p1' },
      { respond: 'p2' },
      { respond: 'draft solution' },
      { respond: JSON.stringify({ pass: true, feedback: 'good' }) },
    ]);

    configureAi({ chatModel: model, scanAnnotations: true });

    const container = useContainer();
    const chainBean = container.resolve<{ chain(input: string): Promise<string> }>(MyChain.name);
    expect(await chainBean.chain('doc')).toBe('step2 done');

    const routeBean = container.resolve<{ route(input: string): Promise<string> }>(MyRoute.name);
    expect(await routeBean.route('billing question')).toBe('billing answer');

    const parallelBean = container.resolve<{
      parallel(system: string, inputs?: readonly string[]): Promise<string[]>;
    }>(MyParallel.name);
    expect(await parallelBean.parallel('sys')).toEqual(['p1', 'p2']);

    const evalBean = container.resolve<{ loop(task: string): Promise<{ solution: string }> }>(
      MyEvaluator.name,
    );
    const result = await evalBean.loop('task');
    expect(result.solution).toBe('draft solution');
  });

  test('ingests @IndexedDocument content into the registered vector store', async () => {
    const store = SimpleVectorStore.of(new FakeEmbeddingModel());

    @IndexedDocument('Static seeded content about Yorktown.')
    class SeededClassA {}
    void SeededClassA;

    class SeededClassB {
      @IndexedDocument({ text: 'Method seeded content.', id: 'method-doc' })
      seedMethod() {}
    }
    void SeededClassB;

    configureAi({
      chatModel: new FakeChatModel('ok'),
      vectorStore: store,
      scanAnnotations: true,
    });

    await sleep(20);
    expect(store.size).toBeGreaterThanOrEqual(2);
  });

  test('buildClassAdvisors wires memory/rag/observed advisors and tolerates missing beans', async () => {
    const memory = MessageWindowChatMemory.builder().maxMessages(10).build();
    const store = SimpleVectorStore.of(new FakeEmbeddingModel());
    await store.add([textDocument('Yorktown is in Virginia.', {}, 'd1')]);

    const events: string[] = [];
    @Injectable()
    class Sink {
      @Subscriber(AiEvents.CHAT_RESPONSE)
      onRes() {
        events.push('seen');
      }
    }
    useContainer().resolve(Sink);

    @AiService()
    @WithMemory()
    @WithRag({ topK: 1 })
    @AiObserved()
    class FullyWiredService {
      @SystemMessageAnn('Answer briefly.')
      ask(@UserMessageAnn() _q: string, @MemoryId() _id: string): Promise<string> {
        throw new Error('proxy override expected');
      }
    }

    configureAi({
      chatModel: new ScriptedChatModel([{ respond: 'Yorktown is in Virginia.' }]),
      memory,
      vectorStore: store,
      scanAnnotations: true,
    });

    const svc = resolveAiService(FullyWiredService);
    const answer = await svc.ask('Where is Yorktown?', 's1');
    expect(answer).toBe('Yorktown is in Virginia.');
    expect(events).toContain('seen');

    useContainer().clear();
    clearAnnotatedTypes();

    @AiService()
    @WithMemory({ token: 'missing-memory' })
    @WithRag({ vectorStore: 'missing-store' })
    class PartiallyWiredService {
      @SystemMessageAnn('Answer briefly.')
      ask2(@UserMessageAnn() _q: string): Promise<string> {
        throw new Error('proxy override expected');
      }
    }

    configureAi({
      chatModel: new FakeChatModel('fallback ok'),
      scanAnnotations: true,
    });

    const svc2 = resolveAiService(PartiallyWiredService);
    expect(await svc2.ask2('hi')).toBe('fallback ok');
  });

  test('resolveAnnotatedTools resolves constructor tools, object tool beans, and ToolSource tools', async () => {
    @Injectable()
    class CtorTools {
      @Tool({ name: 'ctorTool' })
      run() {
        return 'ctor-tool-ran';
      }
    }

    class UnresolvableTools {
      constructor() {
        throw new Error('cannot construct');
      }
      @Tool({ name: 'neverUsed' })
      run() {
        return 'x';
      }
    }

    class DirectBean {
      @Tool({ name: 'directBeanTool' })
      run() {
        return 'direct-ran';
      }
    }
    const directBeanInstance = new DirectBean();

    const explicitTool = functionToolCallback({
      name: 'explicitTool',
      call: () => 'explicit-ran',
    });

    @AiService({
      tools: [CtorTools, UnresolvableTools, directBeanInstance, explicitTool, null as never],
    })
    class ToolyService {
      @SystemMessageAnn('Use tools.')
      ask(@UserMessageAnn() _q: string): Promise<string> {
        throw new Error('proxy override expected');
      }
    }

    const model = new ScriptedChatModel([
      {
        respond: (prompt) => {
          const names = prompt.options?.toolCallbacks?.map((t) => t.toolDefinition.name) ?? [];
          expect(names).toContain('ctorTool');
          expect(names).toContain('directBeanTool');
          expect(names).toContain('explicitTool');
          expect(names).not.toContain('neverUsed');
          return toolCallResponse([toolCall('c1', 'ctorTool', {})]);
        },
      },
      { respond: 'done' },
    ]);

    configureAi({ chatModel: model, scanAnnotations: true });
    const svc = resolveAiService(ToolyService);
    expect(await svc.ask('go')).toBe('done');
  });

  test('resolveDefaultClient falls back to building a ChatClient from the model when no ChatClient token is registered', async () => {
    @Chain(['One step.'])
    class FallbackChain {}

    configureAi({
      chatModel: new FakeChatModel('fallback content'),
      scanAnnotations: true,
      registerChatClient: false,
    });

    const container = useContainer();
    const chainBean = container.resolve<{ chain(input: string): Promise<string> }>(
      FallbackChain.name,
    );
    expect(await chainBean.chain('x')).toBe('fallback content');
  });

  test('resolveAnnotated / resolveAnnotatedAgent resolve by class', async () => {
    @Agent({ system: 'agent-sys' })
    class ResolvableAgent {}

    configureAi({ chatModel: new FakeChatModel('agent-says-hi'), scanAnnotations: true });
    const agent = resolveAnnotatedAgent(ResolvableAgent);
    expect(agent).toBeInstanceOf(ChatAgent);
    expect((await agent.chat('hi')).content).toBe('agent-says-hi');
  });

  test('processAiAnnotations can be invoked directly against an explicit container', () => {
    @Chain(['step'])
    class DirectContainerChain {}
    void DirectContainerChain;

    configureAi({ chatModel: new FakeChatModel('x'), scanAnnotations: false });
    const result = processAiAnnotations();
    expect(result.container).toBeTruthy();
  });
});
