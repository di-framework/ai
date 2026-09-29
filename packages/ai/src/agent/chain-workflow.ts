import type { ChatClient } from '../chat/client/default-chat-client.ts';
import { callChatContent, type WorkflowCallOptions } from './workflow-utils.ts';

export interface ChainStep {
  readonly index: number;
  readonly systemPrompt: string;
  readonly input: string;
  readonly output: string;
}

export interface ChainWorkflowResult {
  readonly result: string;
  readonly steps: readonly ChainStep[];
}

/**
 * Sequential prompt chain: each step’s output feeds the next.
 * Spring AI / Anthropic: Chain Workflow.
 *
 * @example
 * ```ts
 * const chain = new ChainWorkflow(chatClient, [
 *   "Extract key facts from the text.",
 *   "Turn the facts into a short bullet summary.",
 * ]);
 * const summary = await chain.chain(longDocument);
 * ```
 */
export class ChainWorkflow {
  private readonly chatClient: ChatClient;
  private readonly systemPrompts: readonly string[];

  constructor(chatClient: ChatClient, systemPrompts: readonly string[]) {
    if (!systemPrompts.length) {
      throw new Error('ChainWorkflow requires at least one system prompt step');
    }
    this.chatClient = chatClient;
    this.systemPrompts = systemPrompts;
  }

  /** Factory alias (same as {@link chainWorkflow}). */
  static of(chatClient: ChatClient, systemPrompts: readonly string[]): ChainWorkflow {
    return new ChainWorkflow(chatClient, systemPrompts);
  }

  /** Run the chain; return only the final string. */
  async chain(userInput: string, options?: WorkflowCallOptions): Promise<string> {
    const detailed = await this.chainDetailed(userInput, options);
    return detailed.result;
  }

  /** Run the chain and return every intermediate step. */
  async chainDetailed(
    userInput: string,
    options?: WorkflowCallOptions,
  ): Promise<ChainWorkflowResult> {
    let response = userInput;
    const steps: ChainStep[] = [];

    for (let i = 0; i < this.systemPrompts.length; i++) {
      const systemPrompt = this.systemPrompts[i]!;
      const input = `${systemPrompt}\n\n${response}`;
      const output = await callChatContent(this.chatClient, {
        user: input,
        signal: options?.signal,
        options: options?.options,
      });
      steps.push({ index: i, systemPrompt, input: response, output });
      response = output;
    }

    return { result: response, steps };
  }
}

export function chainWorkflow(
  chatClient: ChatClient,
  systemPrompts: readonly string[],
): ChainWorkflow {
  return ChainWorkflow.of(chatClient, systemPrompts);
}
