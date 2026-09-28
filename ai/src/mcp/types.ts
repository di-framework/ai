/**
 * Portable MCP tool / client surfaces used by the adapter layer.
 * Mirrors the JSON shapes from the Model Context Protocol without
 * requiring application code to import the official SDK.
 */

export interface McpConnectionInfo {
  /** Client implementation name (this process). */
  readonly clientName?: string;
  /** Optional human title for the connection (e.g. server alias). */
  readonly title?: string;
  /** Server implementation name when known. */
  readonly serverName?: string;
}

/**
 * Tool descriptor as returned by MCP {@code tools/list}.
 */
export interface McpToolDescriptor {
  readonly name: string;
  readonly description?: string;
  readonly inputSchema?: Record<string, unknown>;
  readonly annotations?: Readonly<Record<string, unknown>>;
  readonly [key: string]: unknown;
}

export interface McpListToolsResult {
  readonly tools: readonly McpToolDescriptor[];
}

export type McpContentBlock =
  | { type: 'text'; text: string; [key: string]: unknown }
  | { type: 'image'; data: string; mimeType: string; [key: string]: unknown }
  | { type: 'audio'; data: string; mimeType: string; [key: string]: unknown }
  | { type: 'resource'; resource: unknown; [key: string]: unknown }
  | { type: string; [key: string]: unknown };

export interface McpCallToolResult {
  readonly content: readonly McpContentBlock[];
  readonly isError?: boolean;
  readonly structuredContent?: unknown;
  readonly [key: string]: unknown;
}

export interface McpCallToolParams {
  readonly name: string;
  readonly arguments?: Record<string, unknown>;
  readonly _meta?: Record<string, unknown>;
}

/**
 * Minimal MCP client session used by {@link McpToolCallback} / provider.
 * Spring AI: {@code McpSyncClient} (async in TypeScript).
 *
 * Adapt the official SDK with {@link adaptSdkClient}, or supply a fake in tests.
 */
export interface McpClientSession {
  readonly connectionInfo?: McpConnectionInfo;
  listTools(): Promise<McpListToolsResult>;
  callTool(params: McpCallToolParams): Promise<McpCallToolResult>;
}

/**
 * Filter discovered tools. Spring AI: {@code McpToolFilter}.
 */
export type McpToolFilter = (connection: McpConnectionInfo, tool: McpToolDescriptor) => boolean;

/**
 * Generate the tool name exposed to the model (may differ from MCP name).
 * Spring AI: {@code McpToolNamePrefixGenerator}.
 */
export type McpToolNamePrefixGenerator = (
  connection: McpConnectionInfo,
  tool: McpToolDescriptor,
) => string;

/**
 * Map {@link ToolContext} into MCP call {@code _meta}.
 * Spring AI: {@code ToolContextToMcpMetaConverter}.
 */
export type ToolContextToMcpMetaConverter = (
  context: ReadonlyMap<string, unknown> | Readonly<Record<string, unknown>>,
) => Record<string, unknown> | undefined;
