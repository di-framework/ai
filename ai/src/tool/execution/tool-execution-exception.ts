import { AiError } from '../../model/errors.ts';
import type { ToolDefinition } from '../definition.ts';

/**
 * Error thrown while executing a tool callback.
 * Spring AI: {@code ToolExecutionException}.
 */
export class ToolExecutionException extends AiError {
  readonly toolDefinition: ToolDefinition;

  constructor(toolDefinition: ToolDefinition, cause: unknown, message?: string) {
    const detail =
      message ??
      (cause instanceof Error ? cause.message : String(cause ?? 'tool execution failed'));
    super(`Tool '${toolDefinition.name}' failed: ${detail}`, 'tool-execution', {
      cause,
      retryable: false,
    });
    this.name = 'ToolExecutionException';
    this.toolDefinition = toolDefinition;
  }
}

export type ToolExecutionExceptionProcessor = (error: ToolExecutionException) => string;

/**
 * Default processor: rethrows. Callers can swap for "return error string to model".
 * Spring AI: {@code DefaultToolExecutionExceptionProcessor} (alwaysThrow default true).
 */
export function defaultToolExecutionExceptionProcessor(
  options: { alwaysThrow?: boolean } = {},
): ToolExecutionExceptionProcessor {
  const alwaysThrow = options.alwaysThrow ?? true;
  return (error) => {
    if (alwaysThrow) throw error;
    return JSON.stringify({
      error: true,
      message: error.message,
      tool: error.toolDefinition.name,
    });
  };
}
