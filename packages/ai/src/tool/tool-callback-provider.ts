import type { ToolCallback } from './tool-callback.ts';

/**
 * Supplies tool callbacks (e.g. from DI or static lists).
 * Spring AI: {@code ToolCallbackProvider}.
 */
export interface ToolCallbackProvider {
  getToolCallbacks(): readonly ToolCallback[];
}

export function staticToolCallbackProvider(
  callbacks: readonly ToolCallback[],
): ToolCallbackProvider {
  const list = [...callbacks];
  return {
    getToolCallbacks: () => list,
  };
}

export function isToolCallbackProvider(value: unknown): value is ToolCallbackProvider {
  return (
    typeof value === 'object' &&
    value != null &&
    typeof (value as ToolCallbackProvider).getToolCallbacks === 'function'
  );
}

export function isToolCallback(value: unknown): value is ToolCallback {
  return (
    typeof value === 'object' &&
    value != null &&
    typeof (value as ToolCallback).call === 'function' &&
    typeof (value as ToolCallback).toolDefinition === 'object' &&
    (value as ToolCallback).toolDefinition != null
  );
}

/**
 * Flatten ToolCallback | ToolCallbackProvider | arrays into a callback list.
 */
export function resolveToolCallbacks(
  ...sources: Array<
    ToolCallback | ToolCallbackProvider | readonly ToolCallback[] | undefined | null
  >
): ToolCallback[] {
  const result: ToolCallback[] = [];
  for (const source of sources) {
    if (source == null) continue;
    if (Array.isArray(source)) {
      result.push(...source);
    } else if (isToolCallbackProvider(source)) {
      result.push(...source.getToolCallbacks());
    } else if (isToolCallback(source)) {
      result.push(source);
    } else {
      throw new Error('Expected ToolCallback, ToolCallbackProvider, or array of ToolCallback');
    }
  }
  validateUniqueToolNames(result);
  return result;
}

export function validateUniqueToolNames(callbacks: readonly ToolCallback[]): void {
  const seen = new Set<string>();
  const duplicates: string[] = [];
  for (const cb of callbacks) {
    const name = cb.toolDefinition.name;
    if (seen.has(name)) duplicates.push(name);
    else seen.add(name);
  }
  if (duplicates.length > 0) {
    throw new Error(`Multiple tools with the same name (${duplicates.join(', ')}) found`);
  }
}

/**
 * Flatten sources and keep the last callback for each tool name
 * (later {@code defaultTools} / bean registrations win).
 */
export function resolveToolCallbacksDedupe(
  ...sources: Array<
    ToolCallback | ToolCallbackProvider | readonly ToolCallback[] | undefined | null
  >
): ToolCallback[] {
  const flat: ToolCallback[] = [];
  for (const source of sources) {
    if (source == null) continue;
    if (Array.isArray(source)) {
      flat.push(...source);
    } else if (isToolCallbackProvider(source)) {
      flat.push(...source.getToolCallbacks());
    } else if (isToolCallback(source)) {
      flat.push(source);
    } else {
      throw new Error('Expected ToolCallback, ToolCallbackProvider, or array of ToolCallback');
    }
  }
  const byName = new Map<string, ToolCallback>();
  for (const cb of flat) {
    byName.set(cb.toolDefinition.name, cb);
  }
  return [...byName.values()];
}
