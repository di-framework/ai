import { AiError } from '../model/errors.ts';
import { defaultResponseTextCleaner, type ResponseTextCleaner } from './response-text-cleaner.ts';
import { NO_JSON_SCHEMA, type StructuredOutputConverter } from './structured-output-converter.ts';

export interface SchemaOutputConverterOptions<T = unknown> {
  /**
   * JSON Schema as a string or plain object.
   * Used for format instructions and optional validation.
   */
  readonly schema?: string | Record<string, unknown>;
  /** Optional display name for format text. */
  readonly name?: string;
  readonly textCleaner?: ResponseTextCleaner;
  /**
   * Map the parsed JSON value to {@code T}. Defaults to identity cast.
   */
  readonly map?: (value: unknown) => T;
  /**
   * Custom format instructions. When omitted, a Spring-like schema instruction is used.
   */
  readonly format?: string;
}

/**
 * Schema-backed structured output converter (TypeScript stand-in for
 * Spring AI {@code BeanOutputConverter}).
 *
 * Pass a JSON Schema; the converter cleans LLM text, {@code JSON.parse}s it,
 * and optionally maps to a typed result.
 */
export class SchemaOutputConverter<T = unknown> implements StructuredOutputConverter<T> {
  private readonly schemaString: string;
  private readonly textCleaner: ResponseTextCleaner;
  private readonly map: (value: unknown) => T;
  private readonly formatOverride?: string;
  private readonly name?: string;

  constructor(options: SchemaOutputConverterOptions<T> = {}) {
    this.schemaString = normalizeSchema(options.schema);
    this.textCleaner = options.textCleaner ?? defaultResponseTextCleaner;
    this.map = options.map ?? ((v) => v as T);
    this.formatOverride = options.format;
    this.name = options.name;
  }

  convert(text: string): T {
    const cleaned = this.textCleaner(text);
    if (!cleaned) {
      throw new AiError('Empty structured output', 'output-validation', {
        retryable: false,
      });
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(cleaned);
    } catch (cause) {
      throw new AiError(
        `Failed to parse structured output as JSON: ${summarize(cleaned)}`,
        'output-validation',
        { cause, retryable: true },
      );
    }
    try {
      return this.map(parsed);
    } catch (cause) {
      throw new AiError(
        `Structured output mapping failed: ${cause instanceof Error ? cause.message : String(cause)}`,
        'output-validation',
        { cause, retryable: false },
      );
    }
  }

  getFormat(): string {
    if (this.formatOverride) return this.formatOverride;
    if (this.schemaString && this.schemaString !== NO_JSON_SCHEMA) {
      return [
        'Your response should be in JSON format.',
        'Do not include any explanations, only provide a RFC8259 compliant JSON response following this format without deviation.',
        'Do not include markdown code blocks in your response.',
        'Remove the ```json markdown from the output.',
        'Here is the JSON Schema instance your output must adhere to:',
        `\`\`\`${this.schemaString}\`\`\``,
      ].join('\n');
    }
    const label = this.name ? ` (${this.name})` : '';
    return [
      `Your response should be in JSON format${label}.`,
      'Do not include any explanations, only provide a RFC8259 compliant JSON response without deviation.',
      'Do not include markdown code blocks in your response.',
    ].join('\n');
  }

  getJsonSchema(): string {
    return this.schemaString;
  }
}

/**
 * Create a {@link SchemaOutputConverter}.
 *
 * @example
 * ```ts
 * const converter = schemaOutputConverter<{ city: string; temp: number }>({
 *   schema: {
 *     type: "object",
 *     properties: {
 *       city: { type: "string" },
 *       temp: { type: "number" },
 *     },
 *     required: ["city", "temp"],
 *   },
 * });
 * ```
 */
export function schemaOutputConverter<T = unknown>(
  options: SchemaOutputConverterOptions<T> = {},
): SchemaOutputConverter<T> {
  return new SchemaOutputConverter(options);
}

/**
 * Convenience: parse JSON object map (no schema required).
 * Spring AI: {@code MapOutputConverter}.
 */
export function mapOutputConverter(
  options: Omit<SchemaOutputConverterOptions<Record<string, unknown>>, 'map'> = {},
): SchemaOutputConverter<Record<string, unknown>> {
  return new SchemaOutputConverter({
    ...options,
    schema:
      options.schema ?? ({ type: 'object', additionalProperties: true } as Record<string, unknown>),
    name: options.name ?? 'Map',
    map: (value) => {
      if (value == null || typeof value !== 'object' || Array.isArray(value)) {
        throw new Error('Expected a JSON object');
      }
      return value as Record<string, unknown>;
    },
    format:
      options.format ??
      [
        'Your response should be in JSON format.',
        'The data structure for the JSON should be a JSON object (map of string keys to values).',
        'Do not include any explanations, only provide a RFC8259 compliant JSON response following this format without deviation.',
        'Remove the ```json markdown surrounding the output including the trailing "```".',
      ].join('\n'),
  });
}

/**
 * Convenience: parse a JSON array.
 * Spring AI: {@code ListOutputConverter} (simplified).
 */
export function listOutputConverter<T = unknown>(
  options: SchemaOutputConverterOptions<T[]> = {},
): SchemaOutputConverter<T[]> {
  return new SchemaOutputConverter({
    ...options,
    schema: options.schema ?? ({ type: 'array' } as Record<string, unknown>),
    name: options.name ?? 'List',
    map: (value) => {
      if (!Array.isArray(value)) {
        throw new Error('Expected a JSON array');
      }
      return options.map ? options.map(value) : (value as T[]);
    },
  });
}

function normalizeSchema(schema?: string | Record<string, unknown>): string {
  if (schema == null) return NO_JSON_SCHEMA;
  if (typeof schema === 'string') {
    const trimmed = schema.trim();
    return trimmed.length > 0 ? trimmed : NO_JSON_SCHEMA;
  }
  return JSON.stringify(schema);
}

function summarize(text: string, max = 120): string {
  const oneLine = text.replace(/\s+/g, ' ').trim();
  return oneLine.length <= max ? oneLine : `${oneLine.slice(0, max)}…`;
}
