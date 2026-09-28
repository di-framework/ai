import type { ToolCallback } from '../../tool/tool-callback.ts';

/**
 * Resolves a tool name to a {@link ToolCallback} when not found in the prompt options.
 * Spring AI: {@code ToolCallbackResolver}.
 */
export interface ToolCallbackResolver {
  resolve(toolName: string): ToolCallback | undefined;
}

export function staticToolCallbackResolver(
  callbacks: readonly ToolCallback[],
): ToolCallbackResolver {
  const byName = new Map(callbacks.map((cb) => [cb.toolDefinition.name, cb] as const));
  return {
    resolve(toolName: string) {
      return byName.get(toolName);
    },
  };
}

export const emptyToolCallbackResolver: ToolCallbackResolver = {
  resolve: () => undefined,
};
