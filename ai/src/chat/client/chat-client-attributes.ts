/**
 * Context keys used by ChatClient / advisors for structured output.
 * Spring AI: {@code ChatClientAttributes}.
 */
export const ChatClientAttributes = {
  OUTPUT_FORMAT: 'outputFormat',
  STRUCTURED_OUTPUT_SCHEMA: 'structuredOutputSchema',
  STRUCTURED_OUTPUT_NATIVE: 'structuredOutputNative',
} as const;

export type ChatClientAttributeKey =
  (typeof ChatClientAttributes)[keyof typeof ChatClientAttributes];
