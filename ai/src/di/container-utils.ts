import { useContainer } from '@di-framework/core/container';
import type { AiContainer, ContainerLike } from './types.ts';

export function asAiContainer(value?: ContainerLike | null): AiContainer {
  return (value ?? useContainer()) as AiContainer;
}

export function registerOnContainer<T>(
  container: AiContainer,
  name: string | (new (...args: never[]) => T) | (abstract new (...args: never[]) => T),
  factory: () => T,
  options: { singleton?: boolean } = { singleton: true },
): void {
  if (typeof container.registerFactory !== 'function') {
    throw new Error('Container does not support registerFactory');
  }
  container.registerFactory(name as string, factory, options);
}

/**
 * Normalize instance-or-factory to a zero-arg factory.
 * Values that look like models (object with {@code call}) are wrapped;
 * plain functions are treated as factories.
 */
export function asFactory<T>(value: T | (() => T)): () => T {
  if (typeof value === 'function' && !isModelLike(value)) {
    return value as () => T;
  }
  return () => value as T;
}

export function isModelLike(value: unknown): boolean {
  return (
    typeof value === 'object' &&
    value != null &&
    typeof (value as { call?: unknown }).call === 'function'
  );
}

export function registerFactoryAliases<T>(
  container: AiContainer,
  factory: () => T,
  tokens: readonly string[],
  singleton = true,
): void {
  const unique = [...new Set(tokens.filter(Boolean))];
  for (const token of unique) {
    registerOnContainer(container, token, factory, { singleton });
  }
}
