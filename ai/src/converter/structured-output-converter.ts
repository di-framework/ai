/**
 * Converts raw LLM text into a structured value, and supplies format
 * instructions for the prompt.
 * Spring AI: {@code StructuredOutputConverter}.
 */
export interface StructuredOutputConverter<T> {
  /** Parse cleaned model text into {@code T}. */
  convert(text: string): T;
  /**
   * Prompt instructions describing the desired format (appended to the user
   * message unless native structured output is used).
   */
  getFormat(): string;
  /**
   * JSON Schema string for the target shape, or empty when unavailable.
   * Spring AI: {@code StructuredOutputConverter.getJsonSchema()}.
   */
  getJsonSchema(): string;
}

/** Spring AI {@code StructuredOutputConverter.NO_JSON_SCHEMA}. */
export const NO_JSON_SCHEMA = '';

export function isStructuredOutputConverter(
  value: unknown,
): value is StructuredOutputConverter<unknown> {
  return (
    typeof value === 'object' &&
    value != null &&
    typeof (value as StructuredOutputConverter<unknown>).convert === 'function' &&
    typeof (value as StructuredOutputConverter<unknown>).getFormat === 'function'
  );
}
