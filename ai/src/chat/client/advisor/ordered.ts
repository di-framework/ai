/**
 * Spring {@code Ordered} precedence constants used by advisors.
 * Lower values run first on the request path (and last on the response path).
 */
export const HIGHEST_PRECEDENCE = Number.MIN_SAFE_INTEGER;
export const LOWEST_PRECEDENCE = Number.MAX_SAFE_INTEGER;

/** Spring AI default for chat-memory advisors. */
export const DEFAULT_CHAT_MEMORY_PRECEDENCE_ORDER = HIGHEST_PRECEDENCE + 200;

/** Spring AI default for {@code ToolCallingAdvisor}. */
export const DEFAULT_TOOL_CALLING_ORDER = HIGHEST_PRECEDENCE + 300;

export function compareOrder(a: { order: number }, b: { order: number }): number {
  return a.order - b.order;
}
