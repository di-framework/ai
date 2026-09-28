import { defineMetadata } from '@di-framework/core/container';
import {
  AI_TOOL_METADATA_KEY,
  getToolMethodMetadata,
  Tool,
  type ToolDecoratorOptions,
  type ToolMethodMetadata,
} from '../tool-decorator.ts';
import { AiAnnKeys } from './keys.ts';
import {
  defineMethodAnn,
  defineOnCtor,
  defineParamAnn,
  readMethodAnnMap,
  readOnCtor,
  readParamAnnMap,
} from './meta.ts';

export interface ToolSetOptions {
  readonly description?: string;
  readonly name?: string;
  readonly auth?: unknown;
}

/** Mark class as a tool group bean (Koog ToolSet). */
export function ToolSet(options: ToolSetOptions | string = {}): ClassDecorator {
  const opts: ToolSetOptions = typeof options === 'string' ? { description: options } : options;
  return (target) => {
    defineOnCtor(AiAnnKeys.TOOL_SET, opts, target as object);
  };
}

/** Alias of {@link ToolSet}. */
export const Tools = ToolSet;

export interface ToolParamOptions {
  readonly description?: string;
  readonly required?: boolean;
  readonly name?: string;
}

/** Describe/require a tool method argument. */
export function ToolParam(options: ToolParamOptions | string = {}): ParameterDecorator {
  const opts: ToolParamOptions = typeof options === 'string' ? { description: options } : options;
  return (target, propertyKey, parameterIndex) => {
    defineParamAnn(
      AiAnnKeys.TOOL_PARAM,
      target as object,
      propertyKey !== undefined ? String(propertyKey) : undefined,
      parameterIndex,
      opts,
    );
  };
}

export interface ToolResultOptions {
  /** Converter id or constructor token resolved later. */
  readonly converter?: string | (new (...args: never[]) => object);
}

/** Custom converter for tool return values. */
export function ToolResult(options: ToolResultOptions = {}): MethodDecorator {
  return (target, propertyKey) => {
    defineMethodAnn(AiAnnKeys.TOOL_RESULT, target as object, String(propertyKey), options);
  };
}

/** Return tool output to the user; skip an extra LLM turn. */
export function ReturnDirect(): MethodDecorator {
  return (target, propertyKey) => {
    const methodName = String(propertyKey);
    defineMethodAnn(AiAnnKeys.RETURN_DIRECT, target as object, methodName, true);
    const ctor =
      typeof target === 'function' ? target : (target as { constructor: object }).constructor;
    const list = [...getToolMethodMetadata(ctor as object)];
    const idx = list.findIndex((m) => m.methodName === methodName);
    if (idx >= 0) {
      const prev = list[idx]!;
      const updated = [...list];
      updated[idx] = { ...prev, returnDirect: true };
      defineMetadata(AI_TOOL_METADATA_KEY, updated, ctor);
    }
  };
}

export function getToolSetOptions(target: object): ToolSetOptions | undefined {
  return readOnCtor(AiAnnKeys.TOOL_SET, target);
}

export function getToolParamAnns(
  target: object,
  methodName: string,
): readonly { index: number; value: ToolParamOptions }[] {
  return readParamAnnMap<ToolParamOptions>(AiAnnKeys.TOOL_PARAM, target)[methodName] ?? [];
}

export function getToolResultOptions(
  target: object,
  methodName: string,
): ToolResultOptions | undefined {
  return readMethodAnnMap<ToolResultOptions>(AiAnnKeys.TOOL_RESULT, target)[methodName];
}

export function isReturnDirect(target: object, methodName: string): boolean {
  return readMethodAnnMap<boolean>(AiAnnKeys.RETURN_DIRECT, target)[methodName] === true;
}

export { Tool, type ToolDecoratorOptions, type ToolMethodMetadata };
