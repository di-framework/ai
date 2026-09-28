import { assistantMessage } from '../../chat/messages/factories.ts';
import type { ChatMessage, ToolResponseMessage } from '../../chat/messages/message.ts';
import { isToolResponseMessage } from '../../chat/messages/message.ts';
import type { Generation } from '../../chat/model/generation.ts';
import { generation } from '../../chat/model/generation.ts';

/** Spring AI {@code ToolExecutionResult.FINISH_REASON}. */
export const TOOL_RETURN_DIRECT_FINISH_REASON = 'returnDirect';

export const TOOL_METADATA_TOOL_ID = 'toolId';
export const TOOL_METADATA_TOOL_NAME = 'toolName';

/**
 * Result of executing model-requested tool calls.
 * Spring AI: {@code ToolExecutionResult}.
 */
export interface ToolExecutionResult {
  /**
   * Conversation history including the assistant tool-call message and tool responses.
   */
  readonly conversationHistory: readonly ChatMessage[];
  /**
   * When true, return tool output to the client instead of calling the model again.
   */
  readonly returnDirect: boolean;
}

export function toolExecutionResult(partial: {
  conversationHistory: readonly ChatMessage[];
  returnDirect?: boolean;
}): ToolExecutionResult {
  return {
    conversationHistory: partial.conversationHistory,
    returnDirect: partial.returnDirect ?? false,
  };
}

/**
 * Build generations from a return-direct tool execution result.
 * Spring AI: {@code ToolExecutionResult.buildGenerations}.
 */
export function buildGenerationsFromToolExecution(result: ToolExecutionResult): Generation[] {
  const history = result.conversationHistory;
  if (history.length === 0) return [];

  const last = history[history.length - 1]!;
  if (!isToolResponseMessage(last)) return [];

  return (last as ToolResponseMessage).responses.map((response) =>
    generation(assistantMessage(response.responseData), {
      finishReason: TOOL_RETURN_DIRECT_FINISH_REASON,
      [TOOL_METADATA_TOOL_ID]: response.id,
      [TOOL_METADATA_TOOL_NAME]: response.name,
    }),
  );
}
