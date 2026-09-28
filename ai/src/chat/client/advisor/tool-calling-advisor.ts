import {
  buildGenerationsFromToolExecution,
  createToolCallingManager,
  defaultToolExecutionEligibilityChecker,
  type ToolCallingManager,
  type ToolExecutionEligibilityChecker,
  type ToolExecutionResult,
} from '../../../model/tool/index.ts';
import type { ChatMessage } from '../../messages/message.ts';
import { ChatResponse } from '../../model/chat-response.ts';
import { TOOL_MANAGER_CONTEXT } from '../../model/tool-execution-context.ts';
import { hasToolCallingOptions } from '../../prompt/chat-options.ts';
import { Prompt } from '../../prompt/prompt.ts';
import { type ChatClientRequest, copyChatClientRequest } from '../chat-client-request.ts';
import { type ChatClientResponse, copyChatClientResponse } from '../chat-client-response.ts';
import type {
  CallAdvisor,
  CallAdvisorChain,
  StreamAdvisor,
  StreamAdvisorChain,
} from './advisor.ts';
import { DEFAULT_TOOL_CALLING_ORDER, HIGHEST_PRECEDENCE, LOWEST_PRECEDENCE } from './ordered.ts';

/**
 * Marker interface for advisors that own the tool-calling loop.
 * Spring AI: {@code ToolAdvisor}.
 */
export interface ToolAdvisor extends CallAdvisor, StreamAdvisor {}

export function isToolAdvisor(advisor: { name: string }): advisor is ToolAdvisor {
  return (
    typeof (advisor as ToolCallingAdvisor).adviseCall === 'function' &&
    (advisor as ToolCallingAdvisor).isToolAdvisor === true
  );
}

export interface ToolCallingAdvisorOptions {
  readonly toolCallingManager?: ToolCallingManager;
  readonly toolExecutionEligibilityChecker?: ToolExecutionEligibilityChecker;
  readonly order?: number;
  /**
   * When false, only the system message + latest tool response are sent on the
   * next model call (expects an outer memory advisor to hold full history).
   * Default true — keep full conversation history inside the loop.
   */
  readonly conversationHistoryEnabled?: boolean;
}

/**
 * Recursive advisor that runs the tool-calling loop outside each {@code ChatModel}.
 * Spring AI: {@code ToolCallingAdvisor}.
 *
 * Order defaults to {@link DEFAULT_TOOL_CALLING_ORDER} so memory advisors (default
 * +200) wrap this advisor and do not re-run on every tool iteration.
 */
export class ToolCallingAdvisor implements ToolAdvisor {
  readonly name = 'Tool Calling Advisor';
  readonly order: number;
  /** Discriminator for {@link isToolAdvisor}. */
  readonly isToolAdvisor = true as const;

  private readonly toolCallingManager: ToolCallingManager;
  private readonly eligibilityChecker: ToolExecutionEligibilityChecker;
  private readonly conversationHistoryEnabled: boolean;

  constructor(options: ToolCallingAdvisorOptions = {}) {
    this.toolCallingManager = options.toolCallingManager ?? createToolCallingManager();
    this.eligibilityChecker =
      options.toolExecutionEligibilityChecker ?? defaultToolExecutionEligibilityChecker;
    this.order = options.order ?? DEFAULT_TOOL_CALLING_ORDER;
    this.conversationHistoryEnabled = options.conversationHistoryEnabled ?? true;

    if (this.order <= HIGHEST_PRECEDENCE || this.order >= LOWEST_PRECEDENCE) {
      throw new Error('advisorOrder must be between HIGHEST_PRECEDENCE and LOWEST_PRECEDENCE');
    }
  }

  static builder(options: ToolCallingAdvisorOptions = {}): ToolCallingAdvisorBuilder {
    return new ToolCallingAdvisorBuilder(options);
  }

  static of(options?: ToolCallingAdvisorOptions): ToolCallingAdvisor {
    return new ToolCallingAdvisor(options);
  }

  async adviseCall(
    request: ChatClientRequest,
    chain: CallAdvisorChain,
  ): Promise<ChatClientResponse> {
    if (!hasToolCallingOptions(request.prompt.options)) {
      return chain.nextCall(request);
    }

    let instructions: readonly ChatMessage[] = request.prompt.messages;
    let chatClientResponse: ChatClientResponse | undefined;
    let isToolCall = false;

    do {
      const processed = copyChatClientRequest(request, {
        prompt: new Prompt(instructions, request.prompt.options),
        context: new Map([...request.context, [TOOL_MANAGER_CONTEXT, this.toolCallingManager]]),
      });

      // Restart advisors after this one (same as Spring chain.copy(this)).
      chatClientResponse = await this.nextCall(chain, processed);

      const chatResponse = chatClientResponse.chatResponse;
      isToolCall = this.eligibilityChecker(chatResponse);

      if (isToolCall && chatResponse) {
        const toolExecutionResult = await this.toolCallingManager.executeToolCalls(
          processed.prompt,
          chatResponse,
        );

        if (toolExecutionResult.returnDirect) {
          return copyChatClientResponse(chatClientResponse, {
            chatResponse: new ChatResponse(
              buildGenerationsFromToolExecution(toolExecutionResult),
              chatResponse.metadata,
            ),
          });
        }

        instructions = this.nextInstructions(processed, toolExecutionResult);
      }
    } while (isToolCall);

    return chatClientResponse!;
  }

  async *adviseStream(
    request: ChatClientRequest,
    chain: StreamAdvisorChain,
  ): AsyncIterable<ChatClientResponse> {
    if (!hasToolCallingOptions(request.prompt.options)) {
      yield* chain.nextStream(request);
      return;
    }

    let instructions: readonly ChatMessage[] = request.prompt.messages;
    let isToolCall = false;

    do {
      const processed = copyChatClientRequest(request, {
        prompt: new Prompt(instructions, request.prompt.options),
        context: new Map([...request.context, [TOOL_MANAGER_CONTEXT, this.toolCallingManager]]),
      });

      const chunks: ChatClientResponse[] = [];
      for await (const chunk of this.nextStream(chain, processed)) {
        chunks.push(chunk);
      }

      const last = chunks[chunks.length - 1];
      const chatResponse = last?.chatResponse;
      isToolCall = this.eligibilityChecker(chatResponse);

      if (isToolCall && chatResponse && last) {
        const toolExecutionResult = await this.toolCallingManager.executeToolCalls(
          processed.prompt,
          chatResponse,
        );

        if (toolExecutionResult.returnDirect) {
          yield copyChatClientResponse(last, {
            chatResponse: new ChatResponse(
              buildGenerationsFromToolExecution(toolExecutionResult),
              chatResponse.metadata,
            ),
          });
          return;
        }

        instructions = this.nextInstructions(processed, toolExecutionResult);
        // Suppress intermediate tool-call stream chunks; continue loop.
      } else {
        for (const chunk of chunks) {
          yield chunk;
        }
      }
    } while (isToolCall);
  }

  private nextInstructions(
    request: ChatClientRequest,
    toolExecutionResult: ToolExecutionResult,
  ): readonly ChatMessage[] {
    if (!this.conversationHistoryEnabled) {
      const history = toolExecutionResult.conversationHistory;
      if (history.length === 0) return history;
      return [request.prompt.getSystemMessage(), history[history.length - 1]!];
    }
    return toolExecutionResult.conversationHistory;
  }

  private nextCall(
    chain: CallAdvisorChain,
    request: ChatClientRequest,
  ): Promise<ChatClientResponse> {
    if (typeof chain.copy === 'function') {
      return chain.copy(this).nextCall(request);
    }
    // Non-mutating index chains can re-enter from the same next position.
    return chain.nextCall(request);
  }

  private nextStream(
    chain: StreamAdvisorChain,
    request: ChatClientRequest,
  ): AsyncIterable<ChatClientResponse> {
    if (typeof chain.copy === 'function') {
      return chain.copy(this).nextStream(request);
    }
    return chain.nextStream(request);
  }
}

export class ToolCallingAdvisorBuilder {
  private options: ToolCallingAdvisorOptions;

  constructor(options: ToolCallingAdvisorOptions = {}) {
    this.options = { ...options };
  }

  toolCallingManager(manager: ToolCallingManager): this {
    this.options = { ...this.options, toolCallingManager: manager };
    return this;
  }

  toolExecutionEligibilityChecker(checker: ToolExecutionEligibilityChecker): this {
    this.options = {
      ...this.options,
      toolExecutionEligibilityChecker: checker,
    };
    return this;
  }

  advisorOrder(order: number): this {
    this.options = { ...this.options, order };
    return this;
  }

  conversationHistoryEnabled(enabled: boolean): this {
    this.options = { ...this.options, conversationHistoryEnabled: enabled };
    return this;
  }

  disableInternalConversationHistory(): this {
    return this.conversationHistoryEnabled(false);
  }

  build(): ToolCallingAdvisor {
    return new ToolCallingAdvisor(this.options);
  }
}
