import type { ChatResponse } from '../../chat/model/chat-response.ts';

/**
 * Determines whether a model response should trigger tool execution.
 * Spring AI: {@code ToolExecutionEligibilityChecker}.
 */
export type ToolExecutionEligibilityChecker = (chatResponse: ChatResponse | undefined) => boolean;

export const defaultToolExecutionEligibilityChecker: ToolExecutionEligibilityChecker = (
  chatResponse,
) => chatResponse?.hasToolCalls() ?? false;
