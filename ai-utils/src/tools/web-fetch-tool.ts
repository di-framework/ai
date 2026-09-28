import { functionToolCallback, type ToolCallback } from '@di-framework/ai';

export interface WebFetchToolOptions {
  readonly timeoutMs?: number;
  readonly maxChars?: number;
}

export interface WebFetchInput {
  readonly url?: string;
}

const DEFAULT_TIMEOUT_MS = 15_000;
const DEFAULT_MAX_CHARS = 80_000;

export function webFetchTool(options: WebFetchToolOptions = {}): ToolCallback {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxChars = options.maxChars ?? DEFAULT_MAX_CHARS;

  return functionToolCallback<WebFetchInput, string>({
    name: 'WebFetch',
    description: `Fetch a public HTTP(S) URL and return text content (truncated).

Do not use for file:// or credentialed private URLs.`,
    inputSchema: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'http or https URL' },
      },
      required: ['url'],
    },
    call: async (input) => {
      const raw = input?.url?.trim() ?? '';
      if (!raw) return 'Error: url is required';
      let parsed: URL;
      try {
        parsed = new URL(raw);
      } catch {
        return `Error: Invalid URL: ${raw}`;
      }
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        return `Error: Only http and https URLs are allowed: ${raw}`;
      }

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const response = await fetch(parsed, { signal: controller.signal, redirect: 'follow' });
        const text = await response.text();
        const body = text.length > maxChars ? `${text.slice(0, maxChars)}\n… (truncated)` : text;
        return `status: ${response.status}\nurl: ${response.url}\n\n${body}`;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return `Error fetching URL: ${message}`;
      } finally {
        clearTimeout(timer);
      }
    },
  });
}
