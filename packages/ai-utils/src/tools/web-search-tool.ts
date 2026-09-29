import { functionToolCallback, type ToolCallback } from '@di-framework/ai';

export interface WebSearchToolOptions {
  readonly apiKey?: string;
  readonly timeoutMs?: number;
}

export interface WebSearchInput {
  readonly query?: string;
  readonly count?: number;
}

export function webSearchTool(options: WebSearchToolOptions = {}): ToolCallback {
  return functionToolCallback<WebSearchInput, string>({
    name: 'WebSearch',
    description: `Search the web (Brave Search API). Requires a Brave API key.`,
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string' },
        count: { type: 'integer', description: 'Number of results (1-10)' },
      },
      required: ['query'],
    },
    call: async (input) => {
      const query = input?.query?.trim() ?? '';
      if (!query) return 'Error: query is required';
      const apiKey = options.apiKey ?? process.env.BRAVE_API_KEY;
      if (!apiKey) {
        return 'Error: Brave API key missing (pass apiKey or set BRAVE_API_KEY)';
      }
      const count = Math.min(10, Math.max(1, input?.count ?? 5));
      const url = new URL('https://api.search.brave.com/res/v1/web/search');
      url.searchParams.set('q', query);
      url.searchParams.set('count', String(count));
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), options.timeoutMs ?? 15_000);
      try {
        const response = await fetch(url, {
          headers: { Accept: 'application/json', 'X-Subscription-Token': apiKey },
          signal: controller.signal,
        });
        const text = await response.text();
        if (!response.ok) {
          return `Error: Brave search failed (${response.status}): ${text.slice(0, 500)}`;
        }
        return formatBraveResults(text);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return `Error searching the web: ${message}`;
      } finally {
        clearTimeout(timer);
      }
    },
  });
}

function formatBraveResults(body: string): string {
  try {
    const parsed = JSON.parse(body) as {
      web?: { results?: { title?: string; url?: string; description?: string }[] };
    };
    const results = parsed.web?.results ?? [];
    if (results.length === 0) return 'No search results';
    return results
      .map(
        (item, index) =>
          `${index + 1}. ${item.title ?? '(untitled)'}\n${item.url ?? ''}\n${item.description ?? ''}`,
      )
      .join('\n\n');
  } catch {
    return body.slice(0, 8000);
  }
}
