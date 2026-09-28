import type { ToolCall } from '../../chat/messages/message.ts';
import type { ToolCallback } from '../../tool/tool-callback.ts';
import type { ToolContext } from '../../tool/tool-context.ts';

/**
 * Context provided to a {@link ToolExecutionAdvisor} during tool execution.
 */
export interface ToolExecutionAdvisorContext {
  readonly toolCall: ToolCall;
  readonly toolCallback: ToolCallback;
  readonly toolContext: ToolContext;
}

/**
 * Invokes the next advisor in the chain or the final tool callback.
 */
export type ToolExecutionAdvisorNext = (context: ToolExecutionAdvisorContext) => Promise<string>;

/**
 * Interceptor/Advisor interface around tool callback execution.
 * Ordered via `order` (lowest number runs first, e.g. HIGHEST_PRECEDENCE first).
 *
 * Spring AI / di-framework alignment: Tool execution advisor contract.
 */
export interface ToolExecutionAdvisor {
  readonly name?: string;
  readonly order?: number;
  adviseExecution(
    context: ToolExecutionAdvisorContext,
    next: ToolExecutionAdvisorNext,
  ): Promise<string>;
}

/** Alias of {@link ToolExecutionAdvisor}. */
export type ToolCallAdvisor = ToolExecutionAdvisor;

/**
 * Execute a chain of {@link ToolExecutionAdvisor}s around a tool callback execution.
 */
export function executeWithAdvisors(
  context: ToolExecutionAdvisorContext,
  advisors: readonly ToolExecutionAdvisor[],
  finalExecution: ToolExecutionAdvisorNext,
): Promise<string> {
  if (!advisors || advisors.length === 0) {
    return finalExecution(context);
  }

  const sorted = [...advisors].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));

  const invoke = (index: number, currentContext: ToolExecutionAdvisorContext): Promise<string> => {
    if (index >= sorted.length) {
      return finalExecution(currentContext);
    }
    const advisor = sorted[index]!;
    return advisor.adviseExecution(currentContext, (nextCtx) => invoke(index + 1, nextCtx));
  };

  return invoke(0, context);
}
