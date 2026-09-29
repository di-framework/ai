import type { AuthorizationManager } from '@di-framework/auth';
import { toolResponse, toolResponseMessage } from '../../chat/messages/factories.ts';
import {
  type AssistantMessage,
  type ChatMessage,
  isAssistantMessage,
  type ToolCall,
} from '../../chat/messages/message.ts';
import type { ChatResponse } from '../../chat/model/chat-response.ts';
import type { ChatOptions } from '../../chat/prompt/chat-options.ts';
import type { Prompt } from '../../chat/prompt/prompt.ts';
import type { ToolDefinition } from '../../tool/definition.ts';
import {
  defaultToolExecutionExceptionProcessor,
  ToolExecutionException,
  type ToolExecutionExceptionProcessor,
} from '../../tool/execution/tool-execution-exception.ts';
import type { ToolCallback } from '../../tool/tool-callback.ts';
import { getToolMetadata } from '../../tool/tool-callback.ts';
import { ToolContext } from '../../tool/tool-context.ts';
import {
  type ToolAuthorizationContext,
  toolAuthorizationAdvisor,
} from './tool-authorization-advisor.ts';
import { emptyToolCallbackResolver, type ToolCallbackResolver } from './tool-callback-resolver.ts';
import type { ToolCallingManager } from './tool-calling-manager.ts';
import {
  executeWithAdvisors,
  type ToolExecutionAdvisor,
  type ToolExecutionAdvisorContext,
} from './tool-execution-advisor.ts';
import { type ToolExecutionResult, toolExecutionResult } from './tool-execution-result.ts';

export interface DefaultToolCallingManagerOptions {
  readonly toolCallbackResolver?: ToolCallbackResolver;
  readonly toolExecutionExceptionProcessor?: ToolExecutionExceptionProcessor;
  /** Ordered list of tool execution advisors. */
  readonly advisors?: readonly ToolExecutionAdvisor[];
  /** Alias for `advisors`. */
  readonly toolExecutionAdvisors?: readonly ToolExecutionAdvisor[];
  /** Explicit AuthorizationManager to automatically wire authorization advisor. */
  readonly authorizationManager?:
    | AuthorizationManager<ToolAuthorizationContext>
    | (() => AuthorizationManager<ToolAuthorizationContext>);
  /** Explicit tool authorization advisor. */
  readonly toolAuthorizationAdvisor?: ToolExecutionAdvisor;
}

/**
 * Create a default {@link ToolCallingManager}.
 * Spring AI: {@code ToolCallingManager.builder().build()}.
 */
export function createToolCallingManager(
  options?: DefaultToolCallingManagerOptions,
): ToolCallingManager {
  return new DefaultToolCallingManager(options);
}

/**
 * Default {@link ToolCallingManager} implementation.
 * Spring AI: {@code DefaultToolCallingManager}.
 */
export class DefaultToolCallingManager implements ToolCallingManager {
  private readonly toolCallbackResolver: ToolCallbackResolver;
  private readonly exceptionProcessor: ToolExecutionExceptionProcessor;
  private readonly advisors: readonly ToolExecutionAdvisor[];

  constructor(options: DefaultToolCallingManagerOptions = {}) {
    this.toolCallbackResolver = options.toolCallbackResolver ?? emptyToolCallbackResolver;
    this.exceptionProcessor =
      options.toolExecutionExceptionProcessor ?? defaultToolExecutionExceptionProcessor();

    const advisors = [...(options.advisors ?? options.toolExecutionAdvisors ?? [])];
    if (options.toolAuthorizationAdvisor) {
      if (!advisors.includes(options.toolAuthorizationAdvisor)) {
        advisors.push(options.toolAuthorizationAdvisor);
      }
    } else if (options.authorizationManager) {
      advisors.push(
        toolAuthorizationAdvisor({ authorizationManager: options.authorizationManager }),
      );
    }
    this.advisors = advisors;
  }

  resolveToolDefinitions(chatOptions: ChatOptions): readonly ToolDefinition[] {
    const callbacks = chatOptions.toolCallbacks ?? [];
    return callbacks.map((cb) => cb.toolDefinition);
  }

  async executeToolCalls(prompt: Prompt, chatResponse: ChatResponse): Promise<ToolExecutionResult> {
    const assistant = findAssistantWithToolCalls(chatResponse);
    if (!assistant) {
      throw new Error('No tool call requested by the chat model');
    }

    const toolContext = buildToolContext(prompt);
    const callbacks = prompt.options?.toolCallbacks ?? [];
    const promptAdvisors: readonly ToolExecutionAdvisor[] =
      (
        prompt.options as {
          toolExecutionAdvisors?: readonly ToolExecutionAdvisor[];
          advisors?: readonly ToolExecutionAdvisor[];
        }
      )?.toolExecutionAdvisors ??
      (
        prompt.options as {
          toolExecutionAdvisors?: readonly ToolExecutionAdvisor[];
          advisors?: readonly ToolExecutionAdvisor[];
        }
      )?.advisors ??
      [];
    const effectiveAdvisors = [...this.advisors, ...promptAdvisors];

    const toolResponses: Array<{
      id: string;
      name: string;
      responseData: string;
    }> = [];

    let returnDirect: boolean | null = null;

    for (const call of assistant.toolCalls) {
      const result = await this.executeOne(call, callbacks, toolContext, effectiveAdvisors);
      toolResponses.push({
        id: call.id,
        name: call.name,
        responseData: result.responseData,
      });
      if (returnDirect === null) {
        returnDirect = result.returnDirect;
      } else {
        returnDirect = returnDirect && result.returnDirect;
      }
    }

    const toolMessage = toolResponseMessage(
      toolResponses.map((r) => toolResponse(r.id, r.name, r.responseData)),
    );

    const conversationHistory: ChatMessage[] = [...prompt.messages, assistant, toolMessage];

    return toolExecutionResult({
      conversationHistory,
      returnDirect: returnDirect ?? false,
    });
  }

  private async executeOne(
    toolCall: ToolCall,
    toolCallbacks: readonly ToolCallback[],
    toolContext: ToolContext,
    advisors: readonly ToolExecutionAdvisor[] = this.advisors,
  ): Promise<{ responseData: string; returnDirect: boolean }> {
    const toolName = toolCall.name;
    const toolCallback =
      toolCallbacks.find((t) => t.toolDefinition.name === toolName) ??
      this.toolCallbackResolver.resolve(toolName);

    if (!toolCallback) {
      throw new Error(`No ToolCallback found for tool name: ${toolName}`);
    }

    const metadata = getToolMetadata(toolCallback);

    const advisorContext: ToolExecutionAdvisorContext = {
      toolCall,
      toolCallback,
      toolContext,
    };

    const finalExecution = async (ctx: ToolExecutionAdvisorContext): Promise<string> => {
      let toolInput = ctx.toolCall.arguments;
      if (!toolInput || toolInput.trim() === '') {
        toolInput = '{}';
      }
      try {
        return await ctx.toolCallback.call(toolInput, ctx.toolContext);
      } catch (error) {
        if (error instanceof ToolExecutionException) {
          return this.exceptionProcessor(error);
        }
        throw error;
      }
    };

    const responseData = await executeWithAdvisors(advisorContext, advisors, finalExecution);

    return {
      responseData: responseData ?? '',
      returnDirect: metadata.returnDirect,
    };
  }
}

function findAssistantWithToolCalls(chatResponse: ChatResponse): AssistantMessage | undefined {
  for (const gen of chatResponse.generations) {
    if (isAssistantMessage(gen.output) && gen.output.toolCalls.length > 0) {
      return gen.output;
    }
  }
  return undefined;
}

function buildToolContext(prompt: Prompt): ToolContext {
  const ctx = prompt.options?.toolContext;
  return new ToolContext(ctx ?? {});
}
