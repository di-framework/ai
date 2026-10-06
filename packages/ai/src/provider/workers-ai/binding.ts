import { AiError } from '../../model/errors.ts';

/**
 * Cloudflare Workers AI binding (`env.AI`).
 * Structural so this package does not depend on `@di-framework/cloudflare`.
 */
export interface WorkersAiBinding {
  run(
    model: string,
    inputs: Record<string, unknown>,
    options?: WorkersAiRunOptions,
  ): Promise<unknown>;
}

export interface WorkersAiRunOptions {
  readonly gateway?: Readonly<Record<string, unknown>>;
  readonly extraHeaders?: Readonly<Record<string, string>>;
  readonly returnRawResponse?: boolean;
}

/**
 * A raw binding, the `@di-framework/cloudflare` `{ binding }` descriptor,
 * or a function that reads the current Worker env. Getters may return `unknown`
 * because `env` is an untyped record.
 */
export type WorkersAiBindingSource =
  | WorkersAiBinding
  | { readonly binding?: unknown }
  | (() => unknown);

export function isWorkersAiBinding(value: unknown): value is WorkersAiBinding {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as WorkersAiBinding).run === 'function'
  );
}

export function resolveWorkersAiBinding(
  source: WorkersAiBindingSource | undefined,
): WorkersAiBinding {
  const resolved = typeof source === 'function' ? source() : source;
  if (isWorkersAiBinding(resolved)) return resolved;
  if (
    resolved &&
    typeof resolved === 'object' &&
    'binding' in resolved &&
    isWorkersAiBinding(resolved.binding)
  ) {
    return resolved.binding;
  }
  throw new AiError(
    'Workers AI binding is not available. Call setCloudflareBindings(env) before the model runs, or pass binding.',
    'invalid-request',
    { provider: 'workers-ai', retryable: false },
  );
}
