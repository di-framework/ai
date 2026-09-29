import type { Container as DIContainer } from '@di-framework/core/container';
import type { Advisor } from '../chat/client/advisor/advisor.ts';
import type { ToolSource } from '../chat/client/default-chat-client.ts';
import type { ChatMemory } from '../chat/memory/chat-memory.ts';
import type { ChatModel } from '../chat/model/chat-model.ts';
import type { ChatOptions } from '../chat/prompt/chat-options.ts';
import type { EmbeddingModel } from '../embedding/embedding-model.ts';
import type { VectorStore } from '../vectorstore/vector-store.ts';

/** Structural container surface used by AI DI helpers (avoids tight coupling). */
export interface AiContainer {
  registerFactory?<T>(
    nameOrClass: string | (new (...args: never[]) => T) | (abstract new (...args: never[]) => T),
    factory: () => T,
    options?: { singleton?: boolean },
  ): unknown;
  resolve<T>(
    key: string | (new (...args: never[]) => T) | (abstract new (...args: never[]) => T),
  ): T;
  emit?(event: string, payload: unknown): unknown;
  clear?(): unknown;
}

export type ContainerLike = AiContainer | DIContainer;

export interface RegisterOptions {
  readonly container?: ContainerLike;
  readonly token?: string;
  /** Extra tokens that resolve to the same factory. */
  readonly aliases?: readonly string[];
  readonly singleton?: boolean;
}

export interface ObservationRegistrationOptions {
  /**
   * When true (default), register an advisor that emits redacted
   * `ai.chat.*` events on the container.
   */
  readonly enabled?: boolean;
  /** Include full prompt message text (default false). */
  readonly includePromptText?: boolean;
  /** Include full response text (default false). */
  readonly includeResponseText?: boolean;
}

export interface ConfigureAiOptions {
  readonly container?: ContainerLike;
  /**
   * Chat model instance or factory.
   * Required unless a model is already registered under {@link chatModelToken}.
   */
  readonly chatModel?: ChatModel | (() => ChatModel);
  /** Token for the chat model. Default {@code chatModel}. */
  readonly chatModelToken?: string;
  /** Also register under {@code chat.default}. Default true. */
  readonly registerChatDefaultAlias?: boolean;
  /** Token for ChatClient. Default {@code chatClient}. */
  readonly chatClientToken?: string;
  /** Token for prototype ChatClient.Builder. Default {@code chatClientBuilder}. */
  readonly chatClientBuilderToken?: string;
  /**
   * When false, do not register the prototype ChatClient.Builder factory.
   * Default true.
   */
  readonly registerChatClientBuilder?: boolean;
  readonly defaultSystem?: string;
  readonly defaultOptions?: ChatOptions;
  readonly tools?: readonly ToolSource[];
  /**
   * Container-managed beans (classes or instances) scanned for `@Tool` methods.
   * Classes are {@code resolve}d from the container.
   */
  readonly toolBeans?: readonly (object | (new (...args: never[]) => object))[];
  readonly memory?: ChatMemory | (() => ChatMemory);
  readonly memoryToken?: string;
  readonly embeddingModel?: EmbeddingModel | (() => EmbeddingModel);
  readonly vectorStore?: VectorStore | (() => VectorStore);
  readonly advisors?: readonly Advisor[];
  readonly observation?: boolean | ObservationRegistrationOptions;
  /**
   * When false, do not register ChatClient factory (model-only setup).
   * Default true.
   */
  readonly registerChatClient?: boolean;
  /**
   * When true (default), scan `@AiService` / `@Agent` / `@ToolSet` / workflow
   * annotations and register factories.
   */
  readonly scanAnnotations?: boolean;
  /**
   * Register a default {@link import("../agent/chat-agent.ts").ChatAgent} under
   * {@link import("./tokens.ts").AiTokens.CHAT_AGENT}.
   */
  readonly agent?: boolean | { readonly system?: string; readonly token?: string };
  /**
   * Opt in to serving A2A HTTP interfaces, task store, and directory of remote origins.
   */
  readonly a2a?:
    | boolean
    | {
        readonly origins?: readonly string[];
        readonly taskStoreToken?: string;
        readonly directoryToken?: string;
        readonly httpHandlerToken?: string;
      };
}

export interface ConfigureAiResult {
  readonly container: ContainerLike;
  readonly chatModelToken: string;
  readonly chatClientToken?: string;
  readonly a2aDirectoryToken?: string;
  readonly a2aHttpHandlerToken?: string;
}
