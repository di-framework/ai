import type { ToolCallback } from '../../tool/tool-callback.ts';

/**
 * Portable chat options, aligned with Spring AI {@code ChatOptions}
 * and tool fields from {@code ToolCallingChatOptions}.
 *
 * Provider-specific options belong in dedicated option types that extend
 * this shape, or in {@link ChatOptions.providerOptions}.
 */
export interface ChatOptions {
  readonly model?: string;
  readonly temperature?: number;
  readonly topP?: number;
  readonly topK?: number;
  readonly frequencyPenalty?: number;
  readonly presencePenalty?: number;
  readonly maxTokens?: number;
  readonly stopSequences?: readonly string[];

  /**
   * TypeScript-native cancellation. Spring uses Reactor/subscription cancellation;
   * we surface {@link AbortSignal} here so providers and the client share one path.
   */
  readonly signal?: AbortSignal;

  /**
   * Tool callbacks available for this request.
   * Spring AI: {@code ToolCallingChatOptions.toolCallbacks}.
   */
  readonly toolCallbacks?: readonly ToolCallback[];

  /**
   * Context map passed into tool callbacks.
   * Spring AI: {@code ToolCallingChatOptions.toolContext}.
   */
  readonly toolContext?: Readonly<Record<string, unknown>>;

  /**
   * Provider-native structured output JSON Schema (when supported).
   * Spring AI: {@code StructuredOutputChatOptions.outputSchema}.
   * Set by ChatClient when {@code useProviderStructuredOutput()} is enabled.
   */
  readonly outputSchema?: string;

  /**
   * Escape hatch for provider-specific features.
   * Core portable code must not interpret these values.
   */
  readonly providerOptions?: Readonly<Record<string, unknown>>;
}

export function chatOptions(partial: ChatOptions = {}): ChatOptions {
  return { ...partial };
}

/**
 * True when options carry tool-calling configuration (callbacks key present).
 * Used by {@code ToolCallingAdvisor} to decide whether to manage the tool loop.
 */
export function hasToolCallingOptions(
  options: ChatOptions | undefined | null,
): options is ChatOptions & { toolCallbacks: readonly ToolCallback[] } {
  return options != null && options.toolCallbacks !== undefined;
}

export function mergeChatOptions(
  base?: ChatOptions | null,
  override?: ChatOptions | null,
): ChatOptions | undefined {
  if (!base && !override) return undefined;
  if (!base) return override ?? undefined;
  if (!override) return base;
  return {
    ...base,
    ...override,
    stopSequences: override.stopSequences ?? base.stopSequences,
    providerOptions: {
      ...base.providerOptions,
      ...override.providerOptions,
    },
    signal: override.signal ?? base.signal,
    // Runtime tools replace defaults entirely when provided (Spring AI merge rule).
    toolCallbacks: override.toolCallbacks ?? base.toolCallbacks,
    toolContext: mergeToolContext(base.toolContext, override.toolContext),
    outputSchema: override.outputSchema ?? base.outputSchema,
  };
}

function mergeToolContext(
  base?: Readonly<Record<string, unknown>>,
  override?: Readonly<Record<string, unknown>>,
): Readonly<Record<string, unknown>> | undefined {
  if (!base && !override) return undefined;
  if (!base) return override;
  if (!override) return base;
  return { ...base, ...override };
}
