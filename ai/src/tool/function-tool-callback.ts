import { type ToolDefinition, toolDefinition } from './definition.ts';
import {
  defaultToolCallResultConverter,
  type ToolCallResultConverter,
} from './execution/tool-call-result-converter.ts';
import { ToolExecutionException } from './execution/tool-execution-exception.ts';
import { type ToolMetadata, toolMetadata } from './metadata.ts';
import type { ToolCallback } from './tool-callback.ts';
import type { ToolContext } from './tool-context.ts';

export type ToolFunction<TInput = unknown, TOutput = unknown> = (
  input: TInput,
  context: ToolContext | undefined,
) => TOutput | Promise<TOutput>;

export interface FunctionToolCallbackOptions<TInput = unknown, TOutput = unknown> {
  readonly name: string;
  readonly description?: string;
  readonly inputSchema?: string | Record<string, unknown>;
  readonly returnDirect?: boolean;
  readonly auth?: unknown;
  readonly toolMetadata?: ToolMetadata;
  readonly resultConverter?: ToolCallResultConverter;
  /**
   * When true (default), JSON-parse the tool input string before calling {@link call}.
   * Set false to receive the raw JSON string as {@code TInput}.
   */
  readonly parseJsonInput?: boolean;
  readonly call: ToolFunction<TInput, TOutput>;
}

/**
 * Function-backed {@link ToolCallback}, aligned with Spring AI {@code FunctionToolCallback}.
 */
export class FunctionToolCallback<TInput = unknown, TOutput = unknown> implements ToolCallback {
  readonly toolDefinition: ToolDefinition;
  readonly toolMetadata: ToolMetadata;
  private readonly parseJsonInput: boolean;
  private readonly fn: ToolFunction<TInput, TOutput>;
  private readonly resultConverter: ToolCallResultConverter;

  constructor(options: FunctionToolCallbackOptions<TInput, TOutput>) {
    this.toolDefinition = toolDefinition({
      name: options.name,
      description: options.description,
      inputSchema: options.inputSchema,
    });
    this.toolMetadata =
      options.toolMetadata ??
      toolMetadata({ returnDirect: options.returnDirect, auth: options.auth });
    this.parseJsonInput = options.parseJsonInput ?? true;
    this.fn = options.call;
    this.resultConverter = options.resultConverter ?? defaultToolCallResultConverter;
  }

  async call(toolInput: string, toolContext?: ToolContext): Promise<string> {
    if (toolInput == null || toolInput === '') {
      throw new Error(`toolInput cannot be null or empty for tool '${this.toolDefinition.name}'`);
    }

    let input: TInput;
    if (this.parseJsonInput) {
      try {
        input = JSON.parse(toolInput) as TInput;
      } catch (cause) {
        throw new ToolExecutionException(
          this.toolDefinition,
          cause,
          `Invalid JSON tool input: ${toolInput}`,
        );
      }
    } else {
      input = toolInput as TInput;
    }

    try {
      const result = await this.fn(input, toolContext);
      return this.resultConverter(result);
    } catch (error) {
      if (error instanceof ToolExecutionException) throw error;
      throw new ToolExecutionException(this.toolDefinition, error);
    }
  }
}

/**
 * Build a {@link FunctionToolCallback}.
 *
 * @example
 * ```ts
 * const weather = functionToolCallback({
 *   name: "getWeather",
 *   description: "Get weather for a city",
 *   inputSchema: {
 *     type: "object",
 *     properties: { city: { type: "string" } },
 *     required: ["city"],
 *   },
 *   call: ({ city }) => ({ temp: 72, city }),
 * });
 * ```
 */
export function functionToolCallback<TInput = unknown, TOutput = unknown>(
  options: FunctionToolCallbackOptions<TInput, TOutput>,
): FunctionToolCallback<TInput, TOutput> {
  return new FunctionToolCallback(options);
}
