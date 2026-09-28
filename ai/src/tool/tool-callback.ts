import type { ToolDefinition } from './definition.ts';
import type { ToolMetadata } from './metadata.ts';
import { DEFAULT_TOOL_METADATA } from './metadata.ts';
import type { ToolContext } from './tool-context.ts';

/**
 * A tool whose execution can be triggered by an AI model.
 * Spring AI: {@code ToolCallback}.
 */
export interface ToolCallback {
  readonly toolDefinition: ToolDefinition;
  readonly toolMetadata?: ToolMetadata;

  /**
   * Execute the tool with a JSON-encoded input string.
   * Returns a string (often JSON) sent back to the model.
   */
  call(toolInput: string, toolContext?: ToolContext): string | Promise<string>;
}

export function getToolMetadata(callback: ToolCallback): ToolMetadata {
  return callback.toolMetadata ?? DEFAULT_TOOL_METADATA;
}
