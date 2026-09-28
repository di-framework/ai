import { A2ADirectory } from '../a2a/directory.ts';
import { A2ATaskStore } from '../a2a/task-store.ts';
import { ChatAgent } from '../agent/chat-agent.ts';
import type { Advisor } from '../chat/client/advisor/advisor.ts';
import { MessageChatMemoryAdvisor } from '../chat/client/advisor/message-chat-memory-advisor.ts';
import type { ChatClient, ChatClientBuilder } from '../chat/client/default-chat-client.ts';
import { ChatClient as ChatClientFactory } from '../chat/client/default-chat-client.ts';
import type { ChatMemory } from '../chat/memory/chat-memory.ts';
import type { ChatModel } from '../chat/model/chat-model.ts';
import type { ToolCallback } from '../tool/tool-callback.ts';
import {
  resolveToolCallbacks,
  staticToolCallbackProvider,
} from '../tool/tool-callback-provider.ts';
import { getAnnotatedTypes } from './annotations/meta.ts';
import { getEnableAiOptions } from './annotations/model.ts';
import { processAiAnnotations } from './annotations/process.ts';
import { asAiContainer, asFactory, registerFactoryAliases } from './container-utils.ts';
import { observationAdvisor } from './observation.ts';
import { AiTokens } from './tokens.ts';
import { toolCallbacksFromBean } from './tool-decorator.ts';
import type {
  ConfigureAiOptions,
  ConfigureAiResult,
  ContainerLike,
  RegisterOptions,
} from './types.ts';

/**
 * Register a {@link ChatModel} under a string token (default {@link AiTokens.CHAT_MODEL}).
 */
export function registerChatModel(
  model: ChatModel | (() => ChatModel),
  options: RegisterOptions = {},
): ChatModel | (() => ChatModel) {
  const container = asAiContainer(options.container);
  const token = options.token ?? AiTokens.CHAT_MODEL;
  const factory = asFactory(model);
  const aliases = options.aliases ?? [];
  registerFactoryAliases(container, factory, [token, ...aliases], options.singleton ?? true);
  return model;
}

/**
 * Register a {@link ChatClient} under a string token (default {@link AiTokens.CHAT_CLIENT}).
 */
export function registerChatClient(
  client: ChatClient | (() => ChatClient),
  options: RegisterOptions = {},
): ChatClient | (() => ChatClient) {
  const container = asAiContainer(options.container);
  const token = options.token ?? AiTokens.CHAT_CLIENT;
  const factory = asFactory(client);
  registerFactoryAliases(
    container,
    factory,
    [token, ...(options.aliases ?? [])],
    options.singleton ?? true,
  );
  return client;
}

/**
 * Register a prototype {@link ChatClientBuilder} factory
 * (default {@link AiTokens.CHAT_CLIENT_BUILDER}, {@code singleton: false}).
 */
export function registerChatClientBuilder(
  factory: () => ChatClientBuilder,
  options: RegisterOptions = {},
): void {
  const container = asAiContainer(options.container);
  const token = options.token ?? AiTokens.CHAT_CLIENT_BUILDER;
  registerFactoryAliases(container, factory, [token, ...(options.aliases ?? [])], false);
}

/**
 * Register chat memory under {@link AiTokens.CHAT_MEMORY} (or custom token).
 */
export function registerChatMemory(
  memory: ChatMemory | (() => ChatMemory),
  options: RegisterOptions = {},
): ChatMemory | (() => ChatMemory) {
  const container = asAiContainer(options.container);
  const token = options.token ?? AiTokens.CHAT_MEMORY;
  registerFactoryAliases(
    container,
    asFactory(memory),
    [token, ...(options.aliases ?? [])],
    options.singleton ?? true,
  );
  return memory;
}

/**
 * Register tool callbacks under {@link AiTokens.TOOL_CALLBACKS}.
 */
export function registerToolCallbacks(
  tools: readonly ToolCallback[],
  options: RegisterOptions = {},
): readonly ToolCallback[] {
  const container = asAiContainer(options.container);
  const token = options.token ?? AiTokens.TOOL_CALLBACKS;
  const list = [...tools];
  registerFactoryAliases(
    container,
    () => list,
    [token, ...(options.aliases ?? [])],
    options.singleton ?? true,
  );
  return list;
}

/**
 * Register a {@link ChatAgent} under {@link AiTokens.CHAT_AGENT} (or custom token).
 */
export function registerChatAgent(
  agent: ChatAgent | (() => ChatAgent),
  options: RegisterOptions = {},
): ChatAgent | (() => ChatAgent) {
  const container = asAiContainer(options.container);
  const token = options.token ?? AiTokens.CHAT_AGENT;
  registerFactoryAliases(
    container,
    asFactory(agent),
    [token, ...(options.aliases ?? [])],
    options.singleton ?? true,
  );
  return agent;
}

/**
 * Resolve the default chat model from the container.
 */
export function resolveChatModel(
  container?: ContainerLike,
  token: string = AiTokens.CHAT_MODEL,
): ChatModel {
  return asAiContainer(container).resolve<ChatModel>(token);
}

/**
 * Resolve the default chat client from the container.
 */
export function resolveChatClient(
  container?: ContainerLike,
  token: string = AiTokens.CHAT_CLIENT,
): ChatClient {
  return asAiContainer(container).resolve<ChatClient>(token);
}

/**
 * Resolve a fresh prototype {@link ChatClientBuilder}.
 */
export function resolveChatClientBuilder(
  container?: ContainerLike,
  token: string = AiTokens.CHAT_CLIENT_BUILDER,
): ChatClientBuilder {
  return asAiContainer(container).resolve<ChatClientBuilder>(token);
}

/**
 * Resolve the default chat agent from the container.
 */
export function resolveChatAgent(
  container?: ContainerLike,
  token: string = AiTokens.CHAT_AGENT,
): ChatAgent {
  return asAiContainer(container).resolve<ChatAgent>(token);
}

/**
 * Spring Boot–style “starter” setup: register model, optional memory/tools,
 * prototype ChatClient.Builder, ChatClient, optional agent, and annotation scan.
 */
export function configureAi(options: ConfigureAiOptions): ConfigureAiResult {
  const container = asAiContainer(options.container);
  const chatModelToken = options.chatModelToken ?? AiTokens.CHAT_MODEL;
  const chatClientToken = options.chatClientToken ?? AiTokens.CHAT_CLIENT;
  const chatClientBuilderToken = options.chatClientBuilderToken ?? AiTokens.CHAT_CLIENT_BUILDER;
  const singleton = true;

  // Merge EnableAi options from annotated app classes when present.
  let merged: ConfigureAiOptions = options;
  for (const ctor of getAnnotatedTypes()) {
    const enable = getEnableAiOptions(ctor);
    if (enable) {
      merged = {
        ...enable,
        ...options,
        scanAnnotations: options.scanAnnotations ?? enable.scanAnnotations ?? true,
      };
    }
  }

  // --- Chat model ---
  if (merged.chatModel) {
    const aliases: string[] = [];
    if (merged.registerChatDefaultAlias !== false) {
      aliases.push(AiTokens.CHAT_MODEL_DEFAULT);
    }
    registerChatModel(merged.chatModel, {
      container,
      token: chatModelToken,
      aliases,
      singleton,
    });
  }

  // --- Memory ---
  if (merged.memory) {
    registerChatMemory(merged.memory, {
      container,
      token: merged.memoryToken ?? AiTokens.CHAT_MEMORY,
      singleton,
    });
  }

  // --- Embedding / vector (optional tokens only) ---
  if (merged.embeddingModel) {
    registerFactoryAliases(
      container,
      asFactory(merged.embeddingModel),
      [AiTokens.EMBEDDING_MODEL],
      singleton,
    );
  }
  if (merged.vectorStore) {
    registerFactoryAliases(
      container,
      asFactory(merged.vectorStore),
      [AiTokens.VECTOR_STORE],
      singleton,
    );
  }

  // --- Tools from beans + explicit sources ---
  const toolCallbacks = collectTools(container, merged);
  if (toolCallbacks.length > 0) {
    registerToolCallbacks(toolCallbacks, { container, singleton });
  }

  const buildBuilder = (): ChatClientBuilder =>
    buildChatClientBuilder(container, chatModelToken, merged, toolCallbacks);

  // --- Prototype ChatClient.Builder ---
  if (merged.registerChatClientBuilder !== false) {
    registerChatClientBuilder(buildBuilder, {
      container,
      token: chatClientBuilderToken,
    });
  }

  // --- ChatClient factory ---
  const registerClient = merged.registerChatClient !== false;
  if (registerClient) {
    registerChatClient(() => buildBuilder().build(), {
      container,
      token: chatClientToken,
      singleton,
    });
  }

  // --- Optional default agent ---
  if (merged.agent) {
    const agentOpts = typeof merged.agent === 'object' ? merged.agent : {};
    registerChatAgent(
      () =>
        ChatAgent.create({
          chatClient: container.resolve<ChatClient>(chatClientToken),
          system: agentOpts.system ?? merged.defaultSystem,
        }),
      { container, token: agentOpts.token ?? AiTokens.CHAT_AGENT, singleton },
    );
  }

  // --- A2A Protocol 1.0 (Task Store, Directory, HTTP Serving) ---
  let a2aDirectoryToken: string | undefined;
  let a2aHttpHandlerToken: string | undefined;

  if (merged.a2a) {
    const a2aOpts = typeof merged.a2a === 'object' ? merged.a2a : {};
    const taskStoreToken = a2aOpts.taskStoreToken ?? AiTokens.A2A_TASK_STORE;
    const directoryToken = a2aOpts.directoryToken ?? AiTokens.A2A_DIRECTORY;
    a2aDirectoryToken = directoryToken;
    a2aHttpHandlerToken = a2aOpts.httpHandlerToken ?? AiTokens.A2A_HTTP_HANDLER;

    registerFactoryAliases(
      container,
      () => A2ATaskStore.create(),
      [taskStoreToken, A2ATaskStore as unknown as string],
      singleton,
    );

    registerFactoryAliases(
      container,
      () => A2ADirectory.create({ origins: a2aOpts.origins }),
      [directoryToken, A2ADirectory as unknown as string],
      singleton,
    );
  }

  // --- Annotation scan ---
  if (merged.scanAnnotations !== false) {
    processAiAnnotations({ container, configure: merged });
  }

  return {
    container,
    chatModelToken,
    chatClientToken: registerClient ? chatClientToken : undefined,
    a2aDirectoryToken,
    a2aHttpHandlerToken,
  };
}

/**
 * Apply `@EnableAi` on an application class: merge options and run {@link configureAi}.
 */
export function enableAi(
  appClass: new (...args: never[]) => object,
  overrides: ConfigureAiOptions = {},
): ConfigureAiResult {
  const enable = getEnableAiOptions(appClass) ?? {};
  return configureAi({ ...enable, ...overrides, scanAnnotations: true });
}

function collectTools(
  container: ReturnType<typeof asAiContainer>,
  options: ConfigureAiOptions,
): ToolCallback[] {
  const fromSources = options.tools ? resolveToolCallbacks(...options.tools) : [];

  const fromBeans: ToolCallback[] = [];
  for (const bean of options.toolBeans ?? []) {
    const instance =
      typeof bean === 'function'
        ? container.resolve<object>(bean as new (...args: never[]) => object)
        : bean;
    fromBeans.push(...toolCallbacksFromBean(instance));
  }

  if (fromSources.length === 0 && fromBeans.length === 0) return [];
  return resolveToolCallbacks(fromSources, fromBeans);
}

function buildChatClientBuilder(
  container: ReturnType<typeof asAiContainer>,
  chatModelToken: string,
  options: ConfigureAiOptions,
  toolCallbacks: readonly ToolCallback[],
): ChatClientBuilder {
  const model = container.resolve<ChatModel>(chatModelToken);
  const advisors: Advisor[] = [...(options.advisors ?? [])];

  if (options.observation === true || typeof options.observation === 'object') {
    const obsOpts = typeof options.observation === 'object' ? options.observation : {};
    if (obsOpts.enabled !== false) {
      advisors.push(
        observationAdvisor({
          container,
          includePromptText: obsOpts.includePromptText,
          includeResponseText: obsOpts.includeResponseText,
        }),
      );
    }
  }

  if (options.memory || options.memoryToken) {
    try {
      const memory = container.resolve<ChatMemory>(options.memoryToken ?? AiTokens.CHAT_MEMORY);
      advisors.push(new MessageChatMemoryAdvisor({ chatMemory: memory }));
    } catch {
      // memory not registered — skip
    }
  }

  let builder = ChatClientFactory.builder(model);
  if (options.defaultSystem) {
    builder = builder.defaultSystem(options.defaultSystem);
  }
  if (options.defaultOptions) {
    builder = builder.defaultOptions(options.defaultOptions);
  }
  if (advisors.length) {
    builder = builder.defaultAdvisors(...advisors);
  }
  if (toolCallbacks.length) {
    builder = builder.defaultTools(staticToolCallbackProvider(toolCallbacks));
  }
  return builder;
}
