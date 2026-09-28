/**
 * Token usage, aligned with Spring AI {@code Usage}.
 */
export interface Usage {
  readonly promptTokens: number;
  readonly completionTokens: number;
  readonly totalTokens: number;
  readonly nativeUsage?: unknown;
  readonly cacheReadInputTokens?: number;
  readonly cacheWriteInputTokens?: number;
}

export function usage(partial: {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  nativeUsage?: unknown;
  cacheReadInputTokens?: number;
  cacheWriteInputTokens?: number;
}): Usage {
  const promptTokens = partial.promptTokens ?? 0;
  const completionTokens = partial.completionTokens ?? 0;
  return {
    promptTokens,
    completionTokens,
    totalTokens: partial.totalTokens ?? promptTokens + completionTokens,
    nativeUsage: partial.nativeUsage,
    cacheReadInputTokens: partial.cacheReadInputTokens,
    cacheWriteInputTokens: partial.cacheWriteInputTokens,
  };
}
