/**
 * Well-known string tokens for AI services in the di-framework container.
 *
 * Prefer these over ad-hoc names so `@Component(AiTokens.CHAT_MODEL)` and
 * `configureAi()` stay consistent across apps.
 */
export const AiTokens = {
  /** Primary / default chat model. */
  CHAT_MODEL: 'chatModel',
  /** Named default model alias (Spring-style `chat.default`). */
  CHAT_MODEL_DEFAULT: 'chat.default',
  /** Primary ChatClient built around the default model. */
  CHAT_CLIENT: 'chatClient',
  /**
   * Prototype {@link import("../chat/client/default-chat-client.ts").ChatClientBuilder}
   * (fresh instance per resolve — Spring `@Scope("prototype")`).
   */
  CHAT_CLIENT_BUILDER: 'chatClientBuilder',
  /** Primary ChatAgent when registered via configureAi / @Agent. */
  CHAT_AGENT: 'chatAgent',
  /** Default chat memory bean. */
  CHAT_MEMORY: 'chatMemory',
  /** Default embedding model. */
  EMBEDDING_MODEL: 'embeddingModel',
  /** Default vector store. */
  VECTOR_STORE: 'vectorStore',
  /** Default document retriever. */
  DOCUMENT_RETRIEVER: 'documentRetriever',
  /**
   * Aggregated {@link import("../tool/tool-callback.ts").ToolCallback}[]
   * discovered from `@Tool` beans / configureAi.
   */
  TOOL_CALLBACKS: 'ai.tools',
  /** Aggregated advisors from `@Advisor` beans. */
  ADVISORS: 'ai.advisors',
  /** MCP client session token. */
  MCP_CLIENT: 'mcpClient',
  /** Default A2A directory token. */
  A2A_DIRECTORY: 'a2aDirectory',
  /** Default A2A task store token. */
  A2A_TASK_STORE: 'a2aTaskStore',
  /** Primary A2A HTTP handler token. */
  A2A_HTTP_HANDLER: 'a2aHttpHandler',
} as const;

export type AiToken = (typeof AiTokens)[keyof typeof AiTokens];
