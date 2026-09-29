import type { ToolCallback } from '../tool/tool-callback.ts';
import type { ToolCallbackProvider } from '../tool/tool-callback-provider.ts';
import { validateUniqueToolNames } from '../tool/tool-callback-provider.ts';
import { McpToolCallback } from './mcp-tool-callback.ts';
import {
  defaultMcpToolNamePrefixGenerator,
  defaultToolContextToMcpMetaConverter,
  emptyConnectionInfo,
} from './mcp-tool-utils.ts';
import type {
  McpClientSession,
  McpConnectionInfo,
  McpToolFilter,
  McpToolNamePrefixGenerator,
  ToolContextToMcpMetaConverter,
} from './types.ts';

export interface McpToolCallbackProviderOptions {
  readonly mcpClients: readonly McpClientSession[];
  readonly toolFilter?: McpToolFilter;
  readonly toolNamePrefixGenerator?: McpToolNamePrefixGenerator;
  readonly toolContextToMcpMetaConverter?: ToolContextToMcpMetaConverter;
  /**
   * When true (default), {@link getToolCallbacks} throws if {@link refresh}
   * has not been called yet. When false, returns an empty list until refresh.
   */
  readonly requireRefresh?: boolean;
}

/**
 * Discovers MCP tools and exposes them as {@link ToolCallback}s.
 * Spring AI: {@code SyncMcpToolCallbackProvider}.
 *
 * Discovery is async ({@link refresh}); {@link getToolCallbacks} returns the
 * last successful discovery (TypeScript adaptation of Spring’s sync list).
 *
 * @example
 * ```ts
 * const provider = new McpToolCallbackProvider({ mcpClients: [session] });
 * await provider.refresh();
 * const client = ChatClient.create(model);
 * await client.prompt().user("…").tools(provider).call().content();
 * ```
 */
export class McpToolCallbackProvider implements ToolCallbackProvider {
  private readonly mcpClients: readonly McpClientSession[];
  private readonly toolFilter: McpToolFilter;
  private readonly toolNamePrefixGenerator: McpToolNamePrefixGenerator;
  private readonly toolContextToMcpMetaConverter: ToolContextToMcpMetaConverter;
  private readonly requireRefresh: boolean;

  private cached: readonly ToolCallback[] = [];
  private ready = false;
  private refreshPromise: Promise<readonly ToolCallback[]> | null = null;

  constructor(options: McpToolCallbackProviderOptions) {
    if (!options.mcpClients?.length) {
      throw new Error('At least one mcpClient is required');
    }
    this.mcpClients = options.mcpClients;
    this.toolFilter = options.toolFilter ?? (() => true);
    this.toolNamePrefixGenerator =
      options.toolNamePrefixGenerator ?? defaultMcpToolNamePrefixGenerator;
    this.toolContextToMcpMetaConverter =
      options.toolContextToMcpMetaConverter ?? defaultToolContextToMcpMetaConverter;
    this.requireRefresh = options.requireRefresh ?? true;
  }

  /**
   * List tools from all sessions and rebuild the callback cache.
   * Safe to call concurrently — in-flight refresh is shared.
   */
  async refresh(): Promise<readonly ToolCallback[]> {
    if (this.refreshPromise) return this.refreshPromise;
    this.refreshPromise = this.doRefresh().finally(() => {
      this.refreshPromise = null;
    });
    return this.refreshPromise;
  }

  /**
   * Cached callbacks from the last {@link refresh}.
   * @throws if never refreshed and {@code requireRefresh} is true
   */
  getToolCallbacks(): readonly ToolCallback[] {
    if (!this.ready) {
      if (this.requireRefresh) {
        throw new Error(
          'McpToolCallbackProvider: call await provider.refresh() before getToolCallbacks()',
        );
      }
      return [];
    }
    return this.cached;
  }

  /** Invalidate cache so the next {@link refresh} re-lists tools. */
  invalidate(): void {
    this.ready = false;
    this.cached = [];
  }

  private async doRefresh(): Promise<readonly ToolCallback[]> {
    const callbacks: ToolCallback[] = [];
    for (const mcpClient of this.mcpClients) {
      const connection = resolveConnection(mcpClient);
      const listed = await mcpClient.listTools();
      for (const tool of listed.tools) {
        if (!this.toolFilter(connection, tool)) continue;
        const prefixed = this.toolNamePrefixGenerator(connection, tool);
        callbacks.push(
          new McpToolCallback({
            mcpClient,
            tool,
            prefixedToolName: prefixed,
            toolContextToMcpMetaConverter: this.toolContextToMcpMetaConverter,
          }),
        );
      }
    }
    validateUniqueToolNames(callbacks);
    this.cached = callbacks;
    this.ready = true;
    return this.cached;
  }
}

function resolveConnection(client: McpClientSession): McpConnectionInfo {
  return client.connectionInfo ?? emptyConnectionInfo();
}

/**
 * Discover tools once and return a static provider (no later refresh).
 */
export async function createMcpToolCallbackProvider(
  options: McpToolCallbackProviderOptions,
): Promise<McpToolCallbackProvider> {
  const provider = new McpToolCallbackProvider(options);
  await provider.refresh();
  return provider;
}

/**
 * Convenience: list + map tools from one or more sessions to callbacks.
 */
export async function mcpToolCallbacks(...mcpClients: McpClientSession[]): Promise<ToolCallback[]> {
  const provider = await createMcpToolCallbackProvider({ mcpClients });
  return [...provider.getToolCallbacks()];
}
