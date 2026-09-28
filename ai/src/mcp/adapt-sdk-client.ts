import type {
  McpCallToolParams,
  McpCallToolResult,
  McpClientSession,
  McpConnectionInfo,
  McpListToolsResult,
  McpToolDescriptor,
} from './types.ts';

/**
 * Structural subset of `@modelcontextprotocol/sdk` {@code Client} used by the adapter.
 * Avoids a hard type import so tests can pass fakes and apps can pass the real Client.
 */
export interface SdkMcpClientLike {
  listTools(
    params?: unknown,
    options?: unknown,
  ): Promise<{
    tools: Array<{
      name: string;
      description?: string;
      inputSchema?: Record<string, unknown>;
      [key: string]: unknown;
    }>;
  }>;
  callTool(
    params: {
      name: string;
      arguments?: Record<string, unknown>;
      _meta?: Record<string, unknown>;
    },
    ...rest: unknown[]
  ): Promise<{
    content?: Array<{ type: string; text?: string; [key: string]: unknown }>;
    isError?: boolean;
    structuredContent?: unknown;
    [key: string]: unknown;
  }>;
  getServerVersion?: () => { name?: string; version?: string } | undefined;
  getClientVersion?: () => { name?: string; version?: string } | undefined;
}

export interface AdaptSdkClientOptions {
  /** Override connection metadata used for tool name prefixing. */
  readonly connectionInfo?: McpConnectionInfo;
  /**
   * Human title for this connection (e.g. config key `"filesystem"`).
   * Combined with server/client name when prefixing.
   */
  readonly title?: string;
}

/**
 * Adapt an official MCP SDK {@code Client} (or compatible) to {@link McpClientSession}.
 *
 * @example
 * ```ts
 * import { Client } from "@modelcontextprotocol/sdk/client/index.js";
 * import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
 * import { adaptSdkClient, createMcpToolCallbackProvider } from "@di-framework/ai";
 *
 * const client = new Client({ name: "app", version: "1.0.0" });
 * await client.connect(new StdioClientTransport({ command: "…", args: [] }));
 * const session = adaptSdkClient(client, { title: "fs" });
 * const tools = await createMcpToolCallbackProvider({ mcpClients: [session] });
 * ```
 */
export function adaptSdkClient(
  client: SdkMcpClientLike,
  options: AdaptSdkClientOptions = {},
): McpClientSession {
  if (!client || typeof client.listTools !== 'function') {
    throw new Error('adaptSdkClient requires a Client with listTools/callTool');
  }

  const connectionInfo: McpConnectionInfo = {
    clientName: options.connectionInfo?.clientName,
    serverName: options.connectionInfo?.serverName,
    title: options.title ?? options.connectionInfo?.title,
  };

  // Lazy-fill names from SDK when available.
  const resolveInfo = (): McpConnectionInfo => {
    const server = client.getServerVersion?.();
    const clientVersion = client.getClientVersion?.();
    return {
      clientName: connectionInfo.clientName ?? clientVersion?.name ?? 'mcp-client',
      serverName: connectionInfo.serverName ?? server?.name,
      title: connectionInfo.title,
    };
  };

  return {
    get connectionInfo() {
      return resolveInfo();
    },
    async listTools(): Promise<McpListToolsResult> {
      const result = await client.listTools();
      const tools: McpToolDescriptor[] = (result.tools ?? []).map((t) => ({
        ...t,
        inputSchema: t.inputSchema as Record<string, unknown> | undefined,
      }));
      return { tools };
    },
    async callTool(params: McpCallToolParams): Promise<McpCallToolResult> {
      const result = await client.callTool({
        name: params.name,
        arguments: params.arguments,
        _meta: params._meta,
      });
      return {
        ...result,
        content: (result.content ?? []) as McpCallToolResult['content'],
        isError: result.isError,
        structuredContent: result.structuredContent,
      };
    },
  };
}
