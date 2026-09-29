import type { ConfigureAiOptions } from '../types.ts';
import { AiAnnKeys } from './keys.ts';
import { defineOnCtor, defineParamAnn, readOnCtor } from './meta.ts';

/** Inject or qualify which chat model bean to use. */
export function ChatModel(
  token: string = 'chatModel',
): ParameterDecorator & PropertyDecorator & ClassDecorator {
  return ((
    target: object,
    propertyKey?: string | symbol,
    indexOrDescriptor?: number | PropertyDescriptor,
  ) => {
    if (typeof indexOrDescriptor === 'number') {
      defineParamAnn(
        AiAnnKeys.CHAT_MODEL,
        target,
        propertyKey !== undefined ? String(propertyKey) : undefined,
        indexOrDescriptor,
        token,
      );
      return;
    }
    if (propertyKey !== undefined) {
      // property: store inject-like hint
      defineOnCtor(`${AiAnnKeys.CHAT_MODEL}:prop:${String(propertyKey)}`, token, target);
      return;
    }
    defineOnCtor(AiAnnKeys.CHAT_MODEL, token, target);
  }) as ParameterDecorator & PropertyDecorator & ClassDecorator;
}

/** Inject a named/prebuilt ChatClient. */
export function ChatClientAnn(
  token: string = 'chatClient',
): ParameterDecorator & PropertyDecorator & ClassDecorator {
  return ((
    target: object,
    propertyKey?: string | symbol,
    indexOrDescriptor?: number | PropertyDescriptor,
  ) => {
    if (typeof indexOrDescriptor === 'number') {
      defineParamAnn(
        AiAnnKeys.CHAT_CLIENT,
        target,
        propertyKey !== undefined ? String(propertyKey) : undefined,
        indexOrDescriptor,
        token,
      );
      return;
    }
    if (propertyKey !== undefined) {
      defineOnCtor(`${AiAnnKeys.CHAT_CLIENT}:prop:${String(propertyKey)}`, token, target);
      return;
    }
    defineOnCtor(AiAnnKeys.CHAT_CLIENT, token, target);
  }) as ParameterDecorator & PropertyDecorator & ClassDecorator;
}

export interface AiConfigurationOptions {
  readonly prefix?: string;
}

/** Class that contributes AI beans/options. */
export function AiConfiguration(options: AiConfigurationOptions = {}): ClassDecorator {
  return (target) => {
    defineOnCtor(AiAnnKeys.AI_CONFIGURATION, options, target as object);
  };
}

export type EnableAiOptions = ConfigureAiOptions & {
  readonly scanAnnotations?: boolean;
};

/** Bootstrap AI scanning + default registrations on an app class. */
export function EnableAi(options: EnableAiOptions = {}): ClassDecorator {
  return (target) => {
    defineOnCtor(AiAnnKeys.ENABLE_AI, { scanAnnotations: true, ...options }, target as object);
  };
}

export interface AiPropertiesOptions {
  readonly prefix?: string;
}

/** Bind ai.* config into a typed options object. */
export function AiProperties(options: AiPropertiesOptions = { prefix: 'ai' }): ClassDecorator {
  return (target) => {
    defineOnCtor(AiAnnKeys.AI_PROPERTIES, options, target as object);
  };
}

export function getEnableAiOptions(target: object): EnableAiOptions | undefined {
  return readOnCtor(AiAnnKeys.ENABLE_AI, target);
}

export function getAiConfigurationOptions(target: object): AiConfigurationOptions | undefined {
  return readOnCtor(AiAnnKeys.AI_CONFIGURATION, target);
}

export function getAiPropertiesOptions(target: object): AiPropertiesOptions | undefined {
  return readOnCtor(AiAnnKeys.AI_PROPERTIES, target);
}

export { ChatClientAnn as ChatClientDecorator };
