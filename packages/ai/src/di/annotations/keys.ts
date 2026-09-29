/** Metadata key namespace for `@di-framework/ai` annotations. */
export const AiAnnKeys = {
  TOOL_SET: 'ai:ann:toolSet',
  TOOL_PARAM: 'ai:ann:toolParam',
  TOOL_RESULT: 'ai:ann:toolResult',
  RETURN_DIRECT: 'ai:ann:returnDirect',
  LLM_DESCRIPTION: 'ai:ann:llmDescription',

  AI_SERVICE: 'ai:ann:aiService',
  AGENT: 'ai:ann:agent',
  SYSTEM_MESSAGE: 'ai:ann:systemMessage',
  USER_MESSAGE: 'ai:ann:userMessage',
  ASSISTANT_MESSAGE: 'ai:ann:assistantMessage',
  PROMPT_VARIABLE: 'ai:ann:promptVariable',
  MEMORY_ID: 'ai:ann:memoryId',
  AGENT_STRATEGY: 'ai:ann:agentStrategy',
  MAX_ITERATIONS: 'ai:ann:maxIterations',

  PROMPT: 'ai:ann:prompt',
  STRUCTURED_OUTPUT: 'ai:ann:structuredOutput',
  OUTPUT_CONVERTER: 'ai:ann:outputConverter',

  CHAT_MODEL: 'ai:ann:chatModel',
  CHAT_CLIENT: 'ai:ann:chatClient',
  AI_CONFIGURATION: 'ai:ann:aiConfiguration',
  ENABLE_AI: 'ai:ann:enableAi',
  AI_PROPERTIES: 'ai:ann:aiProperties',

  ADVISOR: 'ai:ann:advisor',
  ADVISOR_ORDER: 'ai:ann:advisorOrder',
  WITH_MEMORY: 'ai:ann:withMemory',
  WITH_RAG: 'ai:ann:withRag',
  WITH_TOOLS: 'ai:ann:withTools',
  AI_OBSERVED: 'ai:ann:aiObserved',

  VECTOR_STORE: 'ai:ann:vectorStore',
  DOCUMENT: 'ai:ann:document',
  INDEXED_DOCUMENT: 'ai:ann:indexedDocument',
  RETRIEVER: 'ai:ann:retriever',
  CHAT_MEMORY: 'ai:ann:chatMemory',
  EMBEDDING_MODEL: 'ai:ann:embeddingModel',

  MCP_CLIENT: 'ai:ann:mcpClient',
  MCP_TOOL: 'ai:ann:mcpTool',

  CHAIN: 'ai:ann:chain',
  ROUTE: 'ai:ann:route',
  PARALLEL: 'ai:ann:parallel',
  ORCHESTRATOR: 'ai:ann:orchestrator',
  WORKER: 'ai:ann:worker',
  EVALUATE: 'ai:ann:evaluate',
  OPTIMIZE: 'ai:ann:optimize',

  /** Classes registered for annotation processing. */
  REGISTRY: 'ai:ann:registry',
} as const;

export type AiAnnKey = (typeof AiAnnKeys)[keyof typeof AiAnnKeys];

export type Constructor<T = object> = new (...args: never[]) => T;
export type AbstractConstructor<T = object> = abstract new (...args: never[]) => T;
export type AnyConstructor<T = object> = Constructor<T> | AbstractConstructor<T>;
