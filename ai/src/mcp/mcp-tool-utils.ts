import { type ToolDefinition, toolDefinition } from '../tool/definition.ts';
import type {
  McpCallToolResult,
  McpConnectionInfo,
  McpContentBlock,
  McpToolDescriptor,
  McpToolNamePrefixGenerator,
  ToolContextToMcpMetaConverter,
} from './types.ts';

/**
 * Context key for optional MCP exchange / transport metadata.
 * Spring AI: {@code McpToolUtils.TOOL_CONTEXT_MCP_EXCHANGE_KEY}.
 */
export const TOOL_CONTEXT_MCP_EXCHANGE_KEY = 'exchange';

/**
 * Build a portable {@link ToolDefinition} from an MCP tool descriptor.
 */
export function createToolDefinitionFromMcp(
  prefixedName: string,
  tool: McpToolDescriptor,
): ToolDefinition {
  return toolDefinition({
    name: prefixedName,
    description: tool.description ?? tool.name,
    inputSchema: tool.inputSchema ?? { type: 'object', properties: {} },
  });
}

/**
 * Prefixed tool name to reduce collisions across MCP servers.
 * Spring AI: {@code McpToolUtils.prefixedToolName}.
 *
 * Result is at most 64 characters (model-friendly).
 */
export function prefixedToolName(
  prefix: string,
  title: string | undefined | null,
  toolName: string,
): string {
  if (!prefix?.trim() || !toolName?.trim()) {
    throw new Error('Prefix or toolName cannot be null or empty');
  }
  let input = shorten(formatToken(prefix));
  if (title?.trim()) {
    input = `${input}_${formatToken(title)}`;
  }
  input = `${input}_${formatToken(toolName)}`;
  if (input.length > 64) {
    input = input.slice(input.length - 64);
  }
  return input;
}

export function formatToken(input: string): string {
  return input
    .replace(/[^\p{L}\p{N}_-]+/gu, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '');
}

function shorten(input: string): string {
  if (input.length <= 16) return input;
  // Keep start + end for long server names.
  return `${input.slice(0, 8)}_${input.slice(-7)}`;
}

/**
 * Default prefix generator: `{client|server}_{title?}_{tool}`.
 */
export const defaultMcpToolNamePrefixGenerator: McpToolNamePrefixGenerator = (connection, tool) => {
  const prefix = connection.serverName?.trim() || connection.clientName?.trim() || 'mcp';
  return prefixedToolName(prefix, connection.title, tool.name);
};

/**
 * No prefix — expose the original MCP tool name.
 * Spring AI: {@code McpToolNamePrefixGenerator.noPrefix()}.
 */
export const noPrefixMcpToolNameGenerator: McpToolNamePrefixGenerator = (_connection, tool) =>
  tool.name;

/**
 * Convert MCP tool result content into a string for the model.
 * Prefers {@code structuredContent} when present, otherwise serializes text blocks.
 */
export function mcpResultToString(result: McpCallToolResult): string {
  if (result.structuredContent !== undefined) {
    return typeof result.structuredContent === 'string'
      ? result.structuredContent
      : JSON.stringify(result.structuredContent);
  }
  return contentBlocksToString(result.content ?? []);
}

export function contentBlocksToString(blocks: readonly McpContentBlock[]): string {
  if (blocks.length === 0) return '';
  if (blocks.length === 1) {
    return singleBlockToString(blocks[0]!);
  }
  return JSON.stringify(blocks.map(blockToJson));
}

function singleBlockToString(block: McpContentBlock): string {
  if (block.type === 'text' && typeof block.text === 'string') {
    return block.text;
  }
  return JSON.stringify(blockToJson(block));
}

function blockToJson(block: McpContentBlock): unknown {
  if (block.type === 'text') return { type: 'text', text: block.text };
  return block;
}

/**
 * Default: pass the entire tool context record as MCP {@code _meta}
 * (excluding the reserved exchange key).
 */
export const defaultToolContextToMcpMetaConverter: ToolContextToMcpMetaConverter = (context) => {
  const record = context instanceof Map ? Object.fromEntries(context) : { ...context };
  delete record[TOOL_CONTEXT_MCP_EXCHANGE_KEY];
  return Object.keys(record).length > 0 ? record : undefined;
};

export function emptyConnectionInfo(): McpConnectionInfo {
  return {};
}
