import { defineMetadata, getOwnMetadata } from '@di-framework/core/container';
import type { AnyConstructor } from './keys.ts';
import { AiAnnKeys } from './keys.ts';

/** Global list of classes that carry AI annotations and need processing. */
const annotatedTypes: AnyConstructor[] = [];

export function trackAnnotatedType(ctor: AnyConstructor): void {
  if (!annotatedTypes.includes(ctor)) {
    annotatedTypes.push(ctor);
  }
}

export function getAnnotatedTypes(): readonly AnyConstructor[] {
  return annotatedTypes;
}

export function clearAnnotatedTypes(): void {
  annotatedTypes.length = 0;
}

export function getCtor(target: object | AnyConstructor): AnyConstructor {
  if (typeof target === 'function') return target as AnyConstructor;
  return (target as { constructor: AnyConstructor }).constructor;
}

export function defineOnCtor<T>(key: string, value: T, target: object | AnyConstructor): void {
  const ctor = getCtor(target);
  defineMetadata(key, value, ctor);
  if (target !== ctor && typeof target !== 'function') {
    defineMetadata(key, value, target);
  }
  trackAnnotatedType(ctor);
}

export function readOnCtor<T>(key: string, target: object | AnyConstructor): T | undefined {
  const ctor = getCtor(target);
  return (
    (getOwnMetadata(key, ctor) as T | undefined) ??
    (typeof target !== 'function' ? (getOwnMetadata(key, target) as T | undefined) : undefined)
  );
}

export interface MethodAnnMap<T> {
  readonly [methodName: string]: T;
}

export function defineMethodAnn<T>(
  key: string,
  target: object,
  methodName: string,
  value: T,
): void {
  const ctor = getCtor(target);
  const existing: Record<string, T> = {
    ...(getOwnMetadata(key, ctor) as Record<string, T> | undefined),
    ...(getOwnMetadata(key, target) as Record<string, T> | undefined),
  };
  existing[methodName] = value;
  defineMetadata(key, existing, ctor);
  defineMetadata(key, existing, target);
  trackAnnotatedType(ctor);
}

export function readMethodAnnMap<T>(key: string, target: object | AnyConstructor): MethodAnnMap<T> {
  return readOnCtor<MethodAnnMap<T>>(key, target) ?? {};
}

export interface ParamAnnEntry<T> {
  readonly index: number;
  readonly value: T;
}

export function defineParamAnn<T>(
  key: string,
  target: object,
  methodName: string | undefined,
  parameterIndex: number,
  value: T,
): void {
  const ctor = getCtor(target);
  const mapKey = methodName ?? 'constructor';
  const existing: Record<string, ParamAnnEntry<T>[]> = {
    ...(getOwnMetadata(key, ctor) as Record<string, ParamAnnEntry<T>[]> | undefined),
    ...(getOwnMetadata(key, target) as Record<string, ParamAnnEntry<T>[]> | undefined),
  };
  const list = [...(existing[mapKey] ?? [])].filter((e) => e.index !== parameterIndex);
  list.push({ index: parameterIndex, value });
  list.sort((a, b) => a.index - b.index);
  existing[mapKey] = list;
  defineMetadata(key, existing, ctor);
  if (target !== ctor) defineMetadata(key, existing, target);
  trackAnnotatedType(ctor);
}

export function readParamAnnMap<T>(
  key: string,
  target: object | AnyConstructor,
): Readonly<Record<string, readonly ParamAnnEntry<T>[]>> {
  return readOnCtor(key, target) ?? {};
}

/** Mark type for processing without overwriting existing class metadata. */
export function markAnnotated(ctor: AnyConstructor): void {
  defineMetadata(AiAnnKeys.REGISTRY, true, ctor);
  trackAnnotatedType(ctor);
}
