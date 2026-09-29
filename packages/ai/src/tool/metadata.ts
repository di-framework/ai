/**
 * Metadata about tool specification and execution.
 * Spring AI: {@code ToolMetadata}.
 */
export interface ToolMetadata {
  /**
   * When true, tool results are returned directly to the client
   * instead of being fed back to the model.
   */
  readonly returnDirect: boolean;

  /**
   * Opaque authorization metadata (e.g. roles, permissions, scopes, policies).
   */
  readonly auth?: unknown;
}

export function toolMetadata(
  partial: { returnDirect?: boolean; auth?: unknown } = {},
): ToolMetadata {
  return {
    returnDirect: partial.returnDirect ?? false,
    ...(partial.auth !== undefined ? { auth: partial.auth } : {}),
  };
}

export const DEFAULT_TOOL_METADATA: ToolMetadata = Object.freeze({
  returnDirect: false,
});
