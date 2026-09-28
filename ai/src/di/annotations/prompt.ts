import type { StructuredOutputConverter } from '../../converter/structured-output-converter.ts';
import { AiAnnKeys } from './keys.ts';
import {
  defineMethodAnn,
  defineOnCtor,
  defineParamAnn,
  readMethodAnnMap,
  readOnCtor,
} from './meta.ts';

export interface PromptOptions {
  readonly template?: string;
  readonly name?: string;
}

/** Attach inline or named prompt template. */
export function Prompt(
  templateOrOptions: string | PromptOptions,
): MethodDecorator & ClassDecorator {
  const opts: PromptOptions =
    typeof templateOrOptions === 'string' ? { template: templateOrOptions } : templateOrOptions;
  return ((target: object, propertyKey?: string | symbol) => {
    if (propertyKey !== undefined) {
      defineMethodAnn(AiAnnKeys.PROMPT, target, String(propertyKey), opts);
    } else {
      defineOnCtor(AiAnnKeys.PROMPT, opts, target);
    }
  }) as MethodDecorator & ClassDecorator;
}

/** Alias of {@link Prompt}. */
export const PromptTemplate = Prompt;

export interface LLMDescriptionOptions {
  readonly value?: string;
}

/** Human/schema description for types, fields, or params. */
export function LLMDescription(
  text: string,
): ClassDecorator & MethodDecorator & ParameterDecorator {
  return ((
    target: object,
    propertyKey?: string | symbol,
    indexOrDescriptor?: number | PropertyDescriptor,
  ) => {
    if (typeof indexOrDescriptor === 'number') {
      defineParamAnn(
        AiAnnKeys.LLM_DESCRIPTION,
        target,
        propertyKey !== undefined ? String(propertyKey) : undefined,
        indexOrDescriptor,
        text,
      );
      return;
    }
    if (propertyKey !== undefined) {
      defineMethodAnn(AiAnnKeys.LLM_DESCRIPTION, target, String(propertyKey), text);
    } else {
      defineOnCtor(AiAnnKeys.LLM_DESCRIPTION, text, target);
    }
  }) as ClassDecorator & MethodDecorator & ParameterDecorator;
}

export interface StructuredOutputOptions {
  readonly schema?: string | Record<string, unknown>;
  readonly useProviderStructuredOutput?: boolean;
  readonly validateSchema?: boolean;
}

/** Parse method result via JSON schema / entity. */
export function StructuredOutput(
  schemaOrOptions: string | Record<string, unknown> | StructuredOutputOptions,
): MethodDecorator {
  const opts: StructuredOutputOptions =
    typeof schemaOrOptions === 'string' ||
    (typeof schemaOrOptions === 'object' &&
      schemaOrOptions != null &&
      !('schema' in schemaOrOptions) &&
      !('useProviderStructuredOutput' in schemaOrOptions) &&
      !('validateSchema' in schemaOrOptions))
      ? { schema: schemaOrOptions as string | Record<string, unknown> }
      : (schemaOrOptions as StructuredOutputOptions);
  return (target, propertyKey) => {
    defineMethodAnn(AiAnnKeys.STRUCTURED_OUTPUT, target as object, String(propertyKey), opts);
  };
}

export interface OutputConverterOptions {
  readonly converter?: StructuredOutputConverter<unknown> | string;
}

/** Plug a custom structured-output converter. */
export function OutputConverter(
  converterOrOptions: StructuredOutputConverter<unknown> | string | OutputConverterOptions,
): MethodDecorator {
  const opts: OutputConverterOptions =
    typeof converterOrOptions === 'object' &&
    converterOrOptions != null &&
    'converter' in converterOrOptions
      ? converterOrOptions
      : { converter: converterOrOptions as StructuredOutputConverter<unknown> | string };
  return (target, propertyKey) => {
    defineMethodAnn(AiAnnKeys.OUTPUT_CONVERTER, target as object, String(propertyKey), opts);
  };
}

export function getPromptOptions(target: object, methodName?: string): PromptOptions | undefined {
  if (methodName) {
    const m = readMethodAnnMap<PromptOptions>(AiAnnKeys.PROMPT, target)[methodName];
    if (m) return m;
  }
  return readOnCtor(AiAnnKeys.PROMPT, target);
}

export function getLLMDescription(target: object): string | undefined {
  return readOnCtor(AiAnnKeys.LLM_DESCRIPTION, target);
}

export function getStructuredOutput(
  target: object,
  methodName: string,
): StructuredOutputOptions | undefined {
  return readMethodAnnMap<StructuredOutputOptions>(AiAnnKeys.STRUCTURED_OUTPUT, target)[methodName];
}

export function getOutputConverter(
  target: object,
  methodName: string,
): OutputConverterOptions | undefined {
  return readMethodAnnMap<OutputConverterOptions>(AiAnnKeys.OUTPUT_CONVERTER, target)[methodName];
}
