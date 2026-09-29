import type { ChatClient } from '../chat/client/default-chat-client.ts';
import { callChatContent, mapPool, type WorkflowCallOptions } from './workflow-utils.ts';

export interface ParallelizationWorkflowOptions extends WorkflowCallOptions {
  /**
   * Max concurrent ChatClient calls. Defaults to the number of inputs
   * (full parallelism).
   */
  readonly concurrency?: number;
}

/**
 * Fan-out the same system instruction across many independent inputs, then
 * collect results in input order.
 * Spring AI / Anthropic: Parallelization Workflow.
 *
 * @example
 * ```ts
 * const results = await new ParallelizationWorkflow(chatClient).parallel(
 *   "Analyze how market changes will impact this stakeholder group.",
 *   ["Customers: …", "Employees: …", "Investors: …"],
 *   { concurrency: 4 },
 * );
 * ```
 */
export class ParallelizationWorkflow {
  private readonly chatClient: ChatClient;

  constructor(chatClient: ChatClient) {
    this.chatClient = chatClient;
  }

  /** Factory alias (same as {@link parallelizationWorkflow}). */
  static of(chatClient: ChatClient): ParallelizationWorkflow {
    return new ParallelizationWorkflow(chatClient);
  }

  async parallel(
    systemPrompt: string,
    inputs: readonly string[],
    options?: ParallelizationWorkflowOptions,
  ): Promise<string[]> {
    if (inputs.length === 0) return [];
    const concurrency = options?.concurrency ?? inputs.length;
    return mapPool(
      inputs,
      concurrency,
      async (userInput) =>
        callChatContent(this.chatClient, {
          system: systemPrompt,
          user: userInput,
          signal: options?.signal,
          options: options?.options,
        }),
      options?.signal,
    );
  }
}

export function parallelizationWorkflow(chatClient: ChatClient): ParallelizationWorkflow {
  return ParallelizationWorkflow.of(chatClient);
}
