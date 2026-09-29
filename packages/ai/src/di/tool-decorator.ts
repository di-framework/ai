import { defineMetadata, getOwnMetadata } from '@di-framework/core/container';
import {
  type FunctionToolCallbackOptions,
  functionToolCallback,
} from '../tool/function-tool-callback.ts';
import type { ToolCallback } from '../tool/tool-callback.ts';
import {
  staticToolCallbackProvider,
  type ToolCallbackProvider,
} from '../tool/tool-callback-provider.ts';
import type { ToolContext } from '../tool/tool-context.ts';
import { AiAnnKeys } from './annotations/keys.ts';
import { readOnCtor } from './annotations/meta.ts';
import type { ToolSetOptions } from './annotations/tools.ts';

/** Metadata key for `@Tool` method descriptors (di-framework metadata store). */
export const AI_TOOL_METADATA_KEY = 'ai:tools';

export interface ToolMethodMetadata {
  readonly methodName: string;
  readonly name: string;
  readonly description?: string;
  readonly inputSchema?: string | Record<string, unknown>;
  readonly returnDirect?: boolean;
  readonly auth?: unknown;
}

export interface ToolDecoratorOptions {
  /** Tool name exposed to the model. Defaults to the method name. */
  readonly name?: string;
  readonly description?: string;
  readonly inputSchema?: string | Record<string, unknown>;
  readonly returnDirect?: boolean;
  readonly auth?: unknown;
}

/**
 * Marks a method as an AI tool (Spring AI–style `@Tool` on a bean method).
 *
 * The declaring class is typically a `@Container()` bean. Discover tools with
 * {@link toolCallbacksFromBean} / {@link configureAi} `{ toolBeans }`.
 *
 * @example
 * ```ts
 * @Container()
 * class WeatherTools {
 *   @Tool({
 *     description: "Get weather for a city",
 *     inputSchema: {
 *       type: "object",
 *       properties: { city: { type: "string" } },
 *       required: ["city"],
 *     },
 *   })
 *   getWeather({ city }: { city: string }) {
 *     return { temp: 68, city };
 *   }
 * }
 * ```
 */
export function Tool(options: ToolDecoratorOptions | string = {}): MethodDecorator {
  const opts: ToolDecoratorOptions =
    typeof options === 'string' ? { description: options } : options;

  return (target, propertyKey, _descriptor) => {
    const methodName = String(propertyKey);
    const ctor =
      typeof target === 'function' ? target : (target as { constructor: object }).constructor;

    const list: ToolMethodMetadata[] =
      getOwnMetadata(AI_TOOL_METADATA_KEY, ctor) ??
      getOwnMetadata(AI_TOOL_METADATA_KEY, target) ??
      [];

    const entry: ToolMethodMetadata = {
      methodName,
      name: opts.name?.trim() || methodName,
      description: opts.description,
      inputSchema: opts.inputSchema,
      returnDirect: opts.returnDirect,
      auth: opts.auth,
    };

    const next = [...list.filter((m) => m.methodName !== methodName), entry];

    // Store on constructor (class) and prototype for robust lookup.
    defineMetadata(AI_TOOL_METADATA_KEY, next, ctor);
    if (target !== ctor) {
      defineMetadata(AI_TOOL_METADATA_KEY, next, target);
    }
  };
}

/**
 * Read `@Tool` metadata from a class constructor or prototype.
 */
export function getToolMethodMetadata(
  target: object | (new (...args: never[]) => object),
): readonly ToolMethodMetadata[] {
  const ctor =
    typeof target === 'function' ? target : (target as { constructor: object }).constructor;
  return (
    getOwnMetadata(AI_TOOL_METADATA_KEY, ctor) ??
    getOwnMetadata(AI_TOOL_METADATA_KEY, target as object) ??
    []
  );
}

/**
 * Build {@link ToolCallback}s bound to a bean instance for all `@Tool` methods.
 */
export function toolCallbacksFromBean(instance: object): ToolCallback[] {
  if (instance == null || typeof instance !== 'object') {
    throw new Error('toolCallbacksFromBean requires a bean instance');
  }
  const meta = getToolMethodMetadata(instance);
  const toolSetOpts = readOnCtor<ToolSetOptions>(AiAnnKeys.TOOL_SET, instance);
  const toolSetAuth = toolSetOpts?.auth;
  const callbacks: ToolCallback[] = [];

  for (const m of meta) {
    const method = (instance as Record<string, unknown>)[m.methodName];
    if (typeof method !== 'function') {
      throw new Error(
        `@Tool method '${m.methodName}' is missing on ${instance.constructor?.name ?? 'bean'}`,
      );
    }
    const bound = method.bind(instance) as (input: unknown, context?: ToolContext) => unknown;

    let auth = m.auth;
    if (toolSetAuth !== undefined) {
      if (
        auth !== undefined &&
        typeof auth === 'object' &&
        auth !== null &&
        typeof toolSetAuth === 'object' &&
        toolSetAuth !== null
      ) {
        auth = {
          ...(toolSetAuth as Record<string, unknown>),
          ...(auth as Record<string, unknown>),
        };
      } else if (auth === undefined) {
        auth = toolSetAuth;
      }
    }

    const options: FunctionToolCallbackOptions = {
      name: m.name,
      description: m.description,
      inputSchema: m.inputSchema,
      returnDirect: m.returnDirect,
      auth,
      call: (input, context) => bound(input, context),
    };
    callbacks.push(functionToolCallback(options));
  }

  return callbacks;
}

/**
 * Flatten tools from multiple bean instances.
 */
export function toolCallbacksFromBeans(...instances: readonly object[]): ToolCallback[] {
  return instances.flatMap((bean) => toolCallbacksFromBean(bean));
}

/**
 * {@link ToolCallbackProvider} backed by one or more tool beans.
 */
export function toolCallbackProviderFromBeans(
  ...instances: readonly object[]
): ToolCallbackProvider {
  return staticToolCallbackProvider(toolCallbacksFromBeans(...instances));
}

/**
 * True when the target has at least one `@Tool` method.
 */
export function hasToolMethods(target: object | (new (...args: never[]) => object)): boolean {
  return getToolMethodMetadata(target).length > 0;
}
