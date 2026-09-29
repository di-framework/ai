import type { A2AArtifact, A2AMessage, A2ATask, TaskStatus } from './types.ts';

/**
 * Execution context passed to an A2AAgentExecutor during task processing.
 */
export interface A2AExecutionContext {
  readonly task: A2ATask;
  readonly signal?: AbortSignal;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

/**
 * Result returned by an A2AAgentExecutor after processing an incoming message.
 */
export interface A2AExecutionResult {
  readonly status: TaskStatus;
  readonly messages?: readonly A2AMessage[];
  readonly artifacts?: readonly A2AArtifact[];
}

/**
 * Interface implemented by agent runners (such as ChatAgent) to execute A2A tasks.
 *
 * Remote callers see only the resulting Task, Messages, and Artifacts.
 * Internal model prompts, tools, and memory are never exposed.
 */
export interface A2AAgentExecutor {
  execute(
    task: A2ATask,
    message: A2AMessage,
    context: A2AExecutionContext,
  ): Promise<A2AExecutionResult>;
}
