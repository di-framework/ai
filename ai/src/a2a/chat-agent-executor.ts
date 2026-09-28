import type { ChatAgent } from '../agent/chat-agent.ts';
import type { A2AAgentExecutor, A2AExecutionContext, A2AExecutionResult } from './executor.ts';
import { type A2AArtifact, type A2AMessage, type A2ATask, createTextMessage } from './types.ts';

export interface ChatAgentA2AExecutorOptions {
  /** Optional function to extract artifacts from the chat response. */
  readonly artifactExtractor?: (
    content: string,
    task: A2ATask,
  ) => readonly A2AArtifact[] | Promise<readonly A2AArtifact[]>;
  /** Optional default artifact name if automatic artifact generation is enabled. */
  readonly createDefaultArtifact?: boolean;
}

/**
 * Bridges an internal ChatAgent to the A2AAgentExecutor protocol interface.
 *
 * Enforces opacity:
 * Incoming A2AMessage parts are converted to user text prompts.
 * The internal ChatModel / Tool / Memory loop executes entirely inside this process.
 * Outgoing wire representation consists ONLY of completed Task state, Messages, and Artifacts.
 * Internal model prompts, tool definitions, and memory identifiers are never leaked.
 */
export class ChatAgentA2AExecutor implements A2AAgentExecutor {
  private readonly agent: ChatAgent;
  private readonly options: ChatAgentA2AExecutorOptions;

  constructor(agent: ChatAgent, options: ChatAgentA2AExecutorOptions = {}) {
    this.agent = agent;
    this.options = options;
  }

  static create(agent: ChatAgent, options?: ChatAgentA2AExecutorOptions): ChatAgentA2AExecutor {
    return new ChatAgentA2AExecutor(agent, options);
  }

  static of(agent: ChatAgent, options?: ChatAgentA2AExecutorOptions): ChatAgentA2AExecutor {
    return new ChatAgentA2AExecutor(agent, options);
  }

  async execute(
    task: A2ATask,
    message: A2AMessage,
    context: A2AExecutionContext,
  ): Promise<A2AExecutionResult> {
    // 1. Extract prompt text from message parts
    const textParts = message.parts
      .filter((p): p is { kind: 'text'; text: string } => p.kind === 'text')
      .map((p) => p.text);

    const dataParts = message.parts
      .filter((p): p is { kind: 'data'; data: Record<string, unknown> } => p.kind === 'data')
      .map((p) => JSON.stringify(p.data));

    const promptText = [...textParts, ...dataParts].join('\n').trim();

    // 2. Execute the process-local ChatAgent
    const chatResult = await this.agent.chat(promptText, {
      signal: context.signal,
      conversationId: task.contextId ?? task.id,
    });

    const responseContent = chatResult.content;
    const outboundMessage = createTextMessage(responseContent, 'agent');

    // 3. Extract or create artifacts if configured
    let artifacts: A2AArtifact[] = [];
    if (this.options.artifactExtractor) {
      const extracted = await this.options.artifactExtractor(responseContent, task);
      artifacts = [...extracted];
    } else if (this.options.createDefaultArtifact) {
      artifacts.push({
        artifactId: `artifact-${task.id}`,
        name: 'result.txt',
        mimeType: 'text/plain',
        parts: [{ kind: 'text', text: responseContent }],
      });
    }

    return {
      status: {
        state: 'completed',
        timestamp: new Date().toISOString(),
      },
      messages: [outboundMessage],
      artifacts,
    };
  }
}

/**
 * Creates an A2AAgentExecutor from an existing ChatAgent instance.
 */
export function createChatAgentA2AExecutor(
  agent: ChatAgent,
  options?: ChatAgentA2AExecutorOptions,
): A2AAgentExecutor {
  return ChatAgentA2AExecutor.create(agent, options);
}
