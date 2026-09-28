import { AiError } from '../model/errors.ts';
import {
  SchemaOutputConverter,
  type SchemaOutputConverterOptions,
} from './schema-output-converter.ts';

/** Minimal structural representation of Standard Schema v1. */
export interface StandardSchemaV1<TInput = unknown, TOutput = TInput> {
  readonly '~standard': {
    readonly version: 1;
    readonly vendor: string;
    readonly validate: (
      value: TInput,
    ) =>
      | TOutput
      | Promise<TOutput>
      | { readonly issues: readonly StandardSchemaIssue[] }
      | Promise<{ readonly issues: readonly StandardSchemaIssue[] }>;
    readonly jsonSchema?: (options?: {
      target?: string;
    }) => Record<string, unknown> | Promise<Record<string, unknown>>;
  };
}
export interface StandardSchemaIssue {
  readonly message: string;
  readonly path?: readonly (string | number)[];
}

export function isStandardSchema(value: unknown): value is StandardSchemaV1 {
  const standard = (value as { '~standard'?: unknown } | null)?.['~standard'];
  return (
    !!standard &&
    typeof standard === 'object' &&
    (standard as { version?: unknown }).version === 1 &&
    typeof (standard as { validate?: unknown }).validate === 'function'
  );
}

export async function parseStandardSchema<T>(
  schema: StandardSchemaV1<unknown, T>,
  value: unknown,
): Promise<T> {
  const result = await schema['~standard'].validate(value);
  if (result && typeof result === 'object' && 'issues' in result) {
    const issues = (result as { issues: readonly StandardSchemaIssue[] }).issues;
    throw new AiError(
      `Standard Schema validation failed: ${issues.map((i) => i.message).join('; ')}`,
      'output-validation',
      { retryable: false },
    );
  }
  return result as T;
}

export interface StandardSchemaOutputConverterOptions<T>
  extends Omit<SchemaOutputConverterOptions<T>, 'schema' | 'map'> {
  readonly schema: StandardSchemaV1<unknown, T>;
}

/** Structured-output converter backed by any Standard Schema implementation (Zod, Valibot, ArkType, ...). */
export class StandardSchemaOutputConverter<T> extends SchemaOutputConverter<T> {
  private readonly standardSchema: StandardSchemaV1<unknown, T>;
  constructor(options: StandardSchemaOutputConverterOptions<T>) {
    let jsonSchema: Record<string, unknown> | undefined;
    const candidate = options.schema['~standard'].jsonSchema;
    if (candidate) {
      try {
        jsonSchema =
          candidate instanceof Function ? undefined : (candidate as Record<string, unknown>);
      } catch {
        /* optional */
      }
    }
    super({ ...options, schema: jsonSchema, map: (value) => value as T });
    this.standardSchema = options.schema;
  }
  async convertAsync(text: string): Promise<T> {
    const value = super.convert(text);
    return parseStandardSchema(this.standardSchema, value);
  }
}

export function standardSchemaOutputConverter<T>(
  options: StandardSchemaOutputConverterOptions<T>,
): StandardSchemaOutputConverter<T> {
  return new StandardSchemaOutputConverter(options);
}
