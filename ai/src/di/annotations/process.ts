import { AgentCardHelper } from '../../a2a/agent-card.ts';
import { createChatAgentA2AExecutor } from '../../a2a/chat-agent-executor.ts';
import { createA2AHttpHandler } from '../../a2a/http-handler.ts';
import type { A2ATaskStore } from '../../a2a/task-store.ts';
import { ChainWorkflow } from '../../agent/chain-workflow.ts';
import { ChatAgent } from '../../agent/chat-agent.ts';
import { EvaluatorOptimizerWorkflow } from '../../agent/evaluator-optimizer-workflow.ts';
import { ParallelizationWorkflow } from '../../agent/parallelization-workflow.ts';
import { RoutingWorkflow } from '../../agent/routing-workflow.ts';
import type { Advisor } from '../../chat/client/advisor/advisor.ts';
import { MessageChatMemoryAdvisor } from '../../chat/client/advisor/message-chat-memory-advisor.ts';
import {
  ChatClient,
  type ChatClientBuilder,
  type ToolSource,
} from '../../chat/client/default-chat-client.ts';
import { renderTemplate } from '../../chat/client/template.ts';
import { CHAT_MEMORY_CONVERSATION_ID, type ChatMemory } from '../../chat/memory/chat-memory.ts';
import { assistantMessage, systemMessage, userMessage } from '../../chat/messages/factories.ts';
import type { ChatModel } from '../../chat/model/chat-model.ts';
import { schemaOutputConverter } from '../../converter/schema-output-converter.ts';
import type { StructuredOutputConverter } from '../../converter/structured-output-converter.ts';
import { textDocument } from '../../document/document.ts';
import { RetrievalAugmentationAdvisor } from '../../rag/advisor/retrieval-augmentation-advisor.ts';
import { VectorStoreDocumentRetriever } from '../../rag/retrieval/vector-store-document-retriever.ts';
import type { ToolCallback } from '../../tool/tool-callback.ts';
import { resolveToolCallbacks } from '../../tool/tool-callback-provider.ts';
import type { VectorStore } from '../../vectorstore/vector-store.ts';
import { asAiContainer, registerOnContainer } from '../container-utils.ts';
import { observationAdvisor } from '../observation.ts';
import { AiTokens } from '../tokens.ts';
import { hasToolMethods, toolCallbacksFromBean } from '../tool-decorator.ts';
import type { AiContainer, ConfigureAiOptions, ContainerLike } from '../types.ts';
import { getAiObserved, getWithMemory, getWithRag, getWithTools } from './advisor.ts';
import {
  type AgentOptions,
  type AiServiceOptions,
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
} from './assistant.ts';
import type { AnyConstructor } from './keys.ts';
import { AiAnnKeys } from './keys.ts';
import { getAnnotatedTypes, readMethodAnnMap, readParamAnnMap } from './meta.ts';
import { getEnableAiOptions } from './model.ts';
import { getOutputConverter, getPromptOptions, getStructuredOutput } from './prompt.ts';
import { getClassIndexedDocument, getIndexedDocuments } from './rag.ts';
import { getToolSetOptions } from './tools.ts';
import {
  getChainOptions,
  getEvaluateOptions,
  getOptimizeOptions,
  getParallelOptions,
  getRouteOptions,
} from './workflow.ts';

export interface ProcessAiAnnotationsOptions {
  readonly container?: ContainerLike;
  readonly configure?: ConfigureAiOptions;
}

export interface ProcessAiAnnotationsResult {
  readonly container: AiContainer;
  readonly aiServices: readonly AnyConstructor[];
  readonly agents: readonly AnyConstructor[];
  readonly toolSets: readonly AnyConstructor[];
}

/**
 * Scan annotated types and register AiService proxies, Agents, tool sets,
 * workflows, and indexed documents on the container.
 */
export function processAiAnnotations(
  options: ProcessAiAnnotationsOptions = {},
): ProcessAiAnnotationsResult {
  const container = asAiContainer(options.container);
  const types = getAnnotatedTypes();

  const aiServices: AnyConstructor[] = [];
  const agents: AnyConstructor[] = [];
  const toolSets: AnyConstructor[] = [];

  for (const ctor of types) {
    void getEnableAiOptions(ctor);

    if (getToolSetOptions(ctor) || hasToolMethods(ctor)) {
      toolSets.push(ctor);
    }

    if (getAiServiceOptions(ctor)) {
      aiServices.push(ctor);
      registerAiService(container, ctor);
    }

    if (getAgentOptions(ctor)) {
      agents.push(ctor);
      registerAgent(container, ctor, options.configure);
    }

    if (getChainOptions(ctor)) {
      registerChainWorkflow(container, ctor);
    }
    if (getRouteOptions(ctor)) {
      registerRouteWorkflow(container, ctor);
    }
    if (getParallelOptions(ctor)) {
      registerParallelWorkflow(container, ctor);
    }
    if (getEvaluateOptions(ctor) || getOptimizeOptions(ctor)) {
      registerEvaluatorWorkflow(container, ctor);
    }

    void ingestIndexedDocuments(container, ctor);
  }

  return { container, aiServices, agents, toolSets };
}

function registerAiService(container: AiContainer, ctor: AnyConstructor): void {
  const opts = getAiServiceOptions(ctor) ?? {};
  const factory = () => createAiServiceProxy(container, ctor, opts);
  registerFactoryForCtor(container, ctor, factory);
}

function registerAgent(
  container: AiContainer,
  ctor: AnyConstructor,
  configure?: ConfigureAiOptions,
): void {
  const opts = getAgentOptions(ctor) ?? {};
  const factory = () => createAgentFromAnnotations(container, ctor, opts);
  registerFactoryForCtor(container, ctor, factory);

  if (opts.a2a && configure?.a2a) {
    const a2aUrl = typeof opts.a2a === 'object' ? opts.a2a.url : 'http://localhost/a2a';
    const binding = typeof opts.a2a === 'object' ? opts.a2a.binding : 'JSONRPC';

    const handlerFactory = () => {
      const card = AgentCardHelper.create({
        name: opts.name ?? ctor.name,
        description: opts.description,
        version: opts.version,
        skills: opts.skills,
        a2a: { url: a2aUrl, binding },
      });

      const agent = container.resolve<ChatAgent>(ctor as unknown as string);
      const executor = createChatAgentA2AExecutor(agent);
      const taskStore = container.resolve<A2ATaskStore>(AiTokens.A2A_TASK_STORE);

      return createA2AHttpHandler({ card, executor, taskStore });
    };

    registerOnContainer(
      container,
      AiTokens.A2A_HTTP_HANDLER,
      handlerFactory as unknown as () => object,
      {
        singleton: true,
      },
    );
    registerOnContainer(
      container,
      `${ctor.name}A2AHttpHandler`,
      handlerFactory as unknown as () => object,
      {
        singleton: true,
      },
    );
  }
}

function registerFactoryForCtor(
  container: AiContainer,
  ctor: AnyConstructor,
  factory: () => unknown,
): void {
  if (typeof container.registerFactory === 'function') {
    container.registerFactory(
      ctor as unknown as string & (new (...args: never[]) => object),
      factory as () => object,
      {
        singleton: true,
      },
    );
  } else {
    registerOnContainer(container, ctor.name, factory as () => object, { singleton: true });
  }
}

function createAgentFromAnnotations(
  container: AiContainer,
  ctor: AnyConstructor,
  opts: AgentOptions,
): ChatAgent {
  const modelToken = opts.chatModel ?? AiTokens.CHAT_MODEL;
  const model = container.resolve<ChatModel>(modelToken);
  const tools = resolveAnnotatedTools(container, opts.tools, ctor);
  const advisors = buildClassAdvisors(container, ctor);
  void getAgentStrategy(ctor);
  void getMaxIterations(ctor);

  let memory: ChatMemory | undefined;
  const withMem = getWithMemory(ctor);
  const wantMemory = opts.memory === true || (withMem != null && withMem.enabled !== false);
  if (wantMemory) {
    try {
      memory = container.resolve<ChatMemory>(withMem?.token ?? AiTokens.CHAT_MEMORY);
    } catch {
      memory = undefined;
    }
  }

  return ChatAgent.create({
    chatModel: model,
    system: opts.system ?? getSystemMessage(ctor),
    tools: tools.length ? tools : undefined,
    advisors: advisors.length ? advisors : undefined,
    memory,
    defaultConversationId: opts.defaultConversationId,
  });
}

function createAiServiceProxy(
  container: AiContainer,
  ctor: AnyConstructor,
  opts: AiServiceOptions,
): object {
  const client = buildClientForAnnotatedClass(container, ctor, opts);
  const proto = ctor.prototype as Record<string, unknown>;
  const proxy = Object.create(proto) as Record<string, unknown>;

  for (const methodName of collectAiServiceMethods(ctor)) {
    proxy[methodName] = async (...args: unknown[]) =>
      invokeAiServiceMethod(ctor, client, methodName, args);
  }

  return proxy;
}

function collectAiServiceMethods(ctor: AnyConstructor): string[] {
  const names = new Set<string>();
  let proto: object | null = ctor.prototype;
  while (proto && proto !== Object.prototype) {
    for (const key of Object.getOwnPropertyNames(proto)) {
      if (key === 'constructor') continue;
      names.add(key);
    }
    proto = Object.getPrototypeOf(proto);
  }
  for (const key of [
    AiAnnKeys.SYSTEM_MESSAGE,
    AiAnnKeys.USER_MESSAGE,
    AiAnnKeys.ASSISTANT_MESSAGE,
    AiAnnKeys.PROMPT,
    AiAnnKeys.STRUCTURED_OUTPUT,
  ]) {
    for (const name of Object.keys(readMethodAnnMap(key, ctor))) names.add(name);
  }
  for (const key of [AiAnnKeys.USER_MESSAGE, AiAnnKeys.MEMORY_ID, AiAnnKeys.PROMPT_VARIABLE]) {
    for (const name of Object.keys(readParamAnnMap(key, ctor))) names.add(name);
  }
  return [...names];
}

async function invokeAiServiceMethod(
  ctor: AnyConstructor,
  client: ChatClient,
  methodName: string,
  args: unknown[],
): Promise<unknown> {
  const system =
    getSystemMessage(ctor, methodName) ??
    getSystemMessage(ctor) ??
    getAiServiceOptions(ctor)?.system;
  const userMsgAnn = getUserMessageAnn(ctor, methodName);
  const userParams = getUserMessageParams(ctor, methodName);
  const memoryParams = getMemoryIdParams(ctor, methodName);
  const vars = getPromptVariables(ctor, methodName);
  const promptOpts = getPromptOptions(ctor, methodName);
  const assistantFewShot = getAssistantMessage(ctor, methodName);

  const varMap: Record<string, unknown> = {};
  for (const v of vars) {
    const name = v.value.name ?? `arg${v.index}`;
    varMap[name] = args[v.index];
  }

  let userText: string | undefined;
  const userParam = userParams[0];
  if (userParam) {
    userText = String(args[userParam.index] ?? '');
  } else if (userMsgAnn?.template) {
    userText = renderTemplate(userMsgAnn.template, varMap);
  } else if (promptOpts?.template) {
    userText = renderTemplate(promptOpts.template, varMap);
  } else if (args.length > 0 && typeof args[0] === 'string') {
    userText = args[0];
  } else {
    userText = args.map(String).join(' ');
  }

  const systemText = system ? renderTemplate(system, varMap) : undefined;

  let spec = client.prompt();
  if (assistantFewShot) {
    const msgs = [];
    if (systemText) msgs.push(systemMessage(systemText));
    msgs.push(assistantMessage(assistantFewShot));
    msgs.push(userMessage(userText ?? ''));
    spec = spec.messages(msgs);
  } else {
    if (systemText) spec = spec.system(systemText);
    if (userText !== undefined) spec = spec.user(userText);
  }

  const context: Record<string, unknown> = {};
  const memoryParam = memoryParams[0];
  if (memoryParam) {
    const id = args[memoryParam.index];
    if (id != null) context[CHAT_MEMORY_CONVERSATION_ID] = String(id);
  }
  if (Object.keys(context).length) {
    spec = spec.advisorContext(context);
  }

  const structured = getStructuredOutput(ctor, methodName);
  const converterAnn = getOutputConverter(ctor, methodName);
  if (structured?.schema || converterAnn?.converter) {
    let converter: StructuredOutputConverter<unknown> | string | Record<string, unknown>;
    if (converterAnn?.converter && typeof converterAnn.converter !== 'string') {
      converter = converterAnn.converter;
    } else if (structured?.schema) {
      converter = schemaOutputConverter({ schema: structured.schema });
    } else {
      converter = structured?.schema ?? {};
    }
    return spec.call().entity(converter as never, (entitySpec) => {
      if (structured?.useProviderStructuredOutput) entitySpec.useProviderStructuredOutput();
      if (structured?.validateSchema) entitySpec.validateSchema();
    });
  }

  return (await spec.call().content()) ?? '';
}

function buildClientForAnnotatedClass(
  container: AiContainer,
  ctor: AnyConstructor,
  opts: AiServiceOptions,
): ChatClient {
  if (opts.chatClient) {
    return container.resolve<ChatClient>(opts.chatClient);
  }

  // Build from the model (not the prewired prototype builder) so class-level
  // tools/advisors are not duplicated with configureAi defaults.
  const model = container.resolve<ChatModel>(opts.chatModel ?? AiTokens.CHAT_MODEL);
  let builder: ChatClientBuilder = ChatClient.builder(model);

  const system = opts.system ?? getSystemMessage(ctor);
  if (system) builder = builder.defaultSystem(system);

  const tools = resolveAnnotatedTools(
    container,
    [...(opts.tools ?? []), ...(getWithTools(ctor)?.tools ?? [])],
    ctor,
  );
  if (tools.length) builder = builder.defaultTools(...tools);

  const advisors = buildClassAdvisors(container, ctor);
  if (advisors.length) builder = builder.defaultAdvisors(...advisors);

  return builder.build();
}

function resolveAnnotatedTools(
  container: AiContainer,
  tools: readonly (AnyConstructor | object | ToolSource)[] | undefined,
  _ctor: AnyConstructor,
): ToolCallback[] {
  if (!tools?.length) return [];
  const callbacks: ToolCallback[] = [];
  for (const t of tools) {
    if (t == null) continue;
    if (typeof t === 'function') {
      try {
        const instance = container.resolve<object>(t as new (...args: never[]) => object);
        if (hasToolMethods(instance)) {
          callbacks.push(...toolCallbacksFromBean(instance));
          continue;
        }
      } catch {
        // not resolvable
      }
      continue;
    }
    if (typeof t === 'object' && hasToolMethods(t)) {
      callbacks.push(...toolCallbacksFromBean(t));
      continue;
    }
    callbacks.push(...resolveToolCallbacks(t as ToolSource));
  }
  return resolveToolCallbacks(callbacks);
}

function buildClassAdvisors(container: AiContainer, ctor: AnyConstructor): Advisor[] {
  const advisors: Advisor[] = [];

  const withMem = getWithMemory(ctor);
  if (withMem && withMem.enabled !== false) {
    try {
      const memory = container.resolve<ChatMemory>(withMem.token ?? AiTokens.CHAT_MEMORY);
      advisors.push(new MessageChatMemoryAdvisor({ chatMemory: memory }));
    } catch {
      // memory not registered
    }
  }

  const withRag = getWithRag(ctor);
  if (withRag && withRag.enabled !== false) {
    try {
      const store = container.resolve<VectorStore>(withRag.vectorStore ?? AiTokens.VECTOR_STORE);
      advisors.push(
        RetrievalAugmentationAdvisor.builder({
          documentRetriever: VectorStoreDocumentRetriever.builder({
            vectorStore: store,
            topK: withRag.topK ?? 4,
          }),
        }),
      );
    } catch {
      // vector store not registered
    }
  }

  const observed = getAiObserved(ctor);
  if (observed && observed.enabled !== false) {
    advisors.push(
      observationAdvisor({
        container,
        includePromptText: observed.includePromptText,
        includeResponseText: observed.includeResponseText,
      }),
    );
  }

  return advisors;
}

function registerChainWorkflow(container: AiContainer, ctor: AnyConstructor): void {
  const opts = getChainOptions(ctor);
  const steps = opts?.steps ?? [];
  registerOnContainer(
    container,
    ctor.name,
    () => new ChainWorkflow(resolveDefaultClient(container), steps),
    { singleton: true },
  );
}

function registerRouteWorkflow(container: AiContainer, ctor: AnyConstructor): void {
  const opts = getRouteOptions(ctor);
  registerOnContainer(
    container,
    ctor.name,
    () => {
      const workflow = new RoutingWorkflow(resolveDefaultClient(container));
      const routes = opts?.routes ?? {};
      return {
        route: (input: string) => workflow.route(input, routes),
        workflow,
        routes,
      };
    },
    { singleton: true },
  );
}

function registerParallelWorkflow(container: AiContainer, ctor: AnyConstructor): void {
  const opts = getParallelOptions(ctor);
  registerOnContainer(
    container,
    ctor.name,
    () => {
      const workflow = new ParallelizationWorkflow(resolveDefaultClient(container));
      const prompts = opts?.prompts ?? [];
      return {
        parallel: (systemPrompt: string, inputs?: readonly string[]) =>
          workflow.parallel(systemPrompt, inputs ?? prompts),
        workflow,
        prompts,
      };
    },
    { singleton: true },
  );
}

function registerEvaluatorWorkflow(container: AiContainer, ctor: AnyConstructor): void {
  registerOnContainer(
    container,
    ctor.name,
    () => {
      const workflow = new EvaluatorOptimizerWorkflow(resolveDefaultClient(container));
      return {
        loop: (task: string) => workflow.loop(task, { maxIterations: getMaxIterations(ctor) ?? 3 }),
        workflow,
      };
    },
    { singleton: true },
  );
}

function resolveDefaultClient(container: AiContainer): ChatClient {
  try {
    return container.resolve<ChatClient>(AiTokens.CHAT_CLIENT);
  } catch {
    const model = container.resolve<ChatModel>(AiTokens.CHAT_MODEL);
    return ChatClient.create(model);
  }
}

async function ingestIndexedDocuments(container: AiContainer, ctor: AnyConstructor): Promise<void> {
  let store: VectorStore;
  try {
    store = container.resolve<VectorStore>(AiTokens.VECTOR_STORE);
  } catch {
    return;
  }

  const docs = [];
  const classDoc = getClassIndexedDocument(ctor);
  if (classDoc?.text) {
    docs.push(textDocument(classDoc.text, classDoc.metadata ?? {}, classDoc.id));
  }
  for (const [method, opts] of Object.entries(getIndexedDocuments(ctor))) {
    if (opts.text) {
      docs.push(textDocument(opts.text, opts.metadata ?? {}, opts.id ?? `${ctor.name}.${method}`));
    }
  }
  if (docs.length) {
    await store.add(docs);
  }
}

/** Resolve a declarative AiService / Agent by class. */
export function resolveAnnotated<T>(ctor: AnyConstructor<T>, container?: ContainerLike): T {
  return asAiContainer(container).resolve<T>(ctor as unknown as new (...args: never[]) => T);
}

export function resolveAiService<T>(ctor: AnyConstructor<T>, container?: ContainerLike): T {
  return resolveAnnotated(ctor, container);
}

export function resolveAnnotatedAgent(ctor: AnyConstructor, container?: ContainerLike): ChatAgent {
  return resolveAnnotated(ctor, container) as unknown as ChatAgent;
}
