import type { ChatResponse } from '../../chat/model/chat-response.ts';
import type { ChatOptions } from '../../chat/prompt/chat-options.ts';
import type { Prompt } from '../../chat/prompt/prompt.ts';
import type { ToolDefinition } from '../../tool/definition.ts';
import type { ToolExecutionResult } from './tool-execution-result.ts';

/**
 * Manages tool definition resolution and tool call execution.
 * Spring AI: {@code ToolCallingManager}.
 */
export interface ToolCallingManager {
  /**
   * Resolve tool definitions from chat options (for providers that need schemas).
   */
  resolveToolDefinitions(chatOptions: ChatOptions): readonly ToolDefinition[];

  /**
   * Execute tool calls requested in {@link chatResponse}, appending results to history.
   */
  executeToolCalls(prompt: Prompt, chatResponse: ChatResponse): Promise<ToolExecutionResult>;
}
