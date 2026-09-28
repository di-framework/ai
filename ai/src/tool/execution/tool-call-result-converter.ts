/**
 * Converts a tool function return value to the string sent back to the model.
 * Spring AI: {@code ToolCallResultConverter}.
 */
export type ToolCallResultConverter = (result: unknown) => string;

export const defaultToolCallResultConverter: ToolCallResultConverter = (result) => {
  if (result == null) return '';
  if (typeof result === 'string') return result;
  try {
    return JSON.stringify(result);
  } catch {
    return String(result);
  }
};
