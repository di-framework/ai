/**
 * Preprocesses LLM text before structured parsing.
 * Spring AI: {@code ResponseTextCleaner}.
 */
export type ResponseTextCleaner = (text: string) => string;

export function whitespaceCleaner(text: string): string {
  return text.trim();
}

/**
 * Removes markdown fenced code blocks (```json … ``` / ``` … ```).
 * Spring AI: {@code MarkdownCodeBlockCleaner}.
 */
export function markdownCodeBlockCleaner(text: string): string {
  let cleaned = text.trim();
  if (!cleaned.startsWith('```') || !cleaned.endsWith('```')) {
    return cleaned;
  }

  const lines = cleaned.split('\n');
  if (lines.length === 1) {
    // ```{"key":1}```
    cleaned = cleaned.slice(3, -3).trim();
    // Drop optional language tag on same line (rare)
    const langMatch = cleaned.match(/^[a-zA-Z0-9_-]+\s+/);
    if (langMatch && cleaned.slice(langMatch[0].length).startsWith('{')) {
      cleaned = cleaned.slice(langMatch[0].length);
    }
    return cleaned.trim();
  }

  // Drop opening fence line and trailing ```
  lines.shift();
  const last = lines[lines.length - 1] ?? '';
  if (last.trim() === '```' || last.trim().endsWith('```')) {
    lines[lines.length - 1] = last.replace(/```\s*$/, '');
  }
  return lines.join('\n').trim();
}

/**
 * Strips common thinking / reasoning tags (e.g. {@code <thinking>…</thinking>}).
 * Spring AI: {@code ThinkingTagCleaner}.
 */
export function thinkingTagCleaner(text: string): string {
  return text
    .replace(/<thinking\b[^>]*>[\s\S]*?<\/thinking>/gi, '')
    .replace(/<think\b[^>]*>[\s\S]*?<\/think>/gi, '')
    .trim();
}

export function compositeResponseTextCleaner(
  ...cleaners: ResponseTextCleaner[]
): ResponseTextCleaner {
  return (text) => cleaners.reduce((acc, cleaner) => cleaner(acc), text);
}

/** Default cleaner pipeline used by schema/map converters. */
export const defaultResponseTextCleaner: ResponseTextCleaner = compositeResponseTextCleaner(
  whitespaceCleaner,
  thinkingTagCleaner,
  markdownCodeBlockCleaner,
  whitespaceCleaner,
);
