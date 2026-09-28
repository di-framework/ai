import type { ToolCallback } from '../tool/tool-callback.ts';
import { getToolMetadata } from '../tool/tool-callback.ts';
import type { McpCallToolResult, McpToolDescriptor } from './types.ts';

/**
 * Describe a local {@link ToolCallback} as an MCP tool descriptor
 * (for registering tools on an MCP server).
 * Spring AI: parts of {@code McpToolUtils} (callback → MCP tool).
 */
export function toolCallbackToMcpDescriptor(callback: ToolCallback): McpToolDescriptor {
  const def = callback.toolDefinition;
  let inputSchema: Record<string, unknown> = {
    type: 'object',
    properties: {},
  };
  try {
    inputSchema = JSON.parse(def.inputSchema) as Record<string, unknown>;
  } catch {
    // keep default
  }
  return {
    name: def.name,
    description: def.description,
    inputSchema,
  };
}

/**
 * Handler shape accepted by many MCP server APIs for tool execution.
 */
export type McpToolHandler = (args: Record<string, unknown>) => Promise<McpCallToolResult>;

/**
 * Wrap a {@link ToolCallback} as an MCP tool handler
 * (JSON-in / content-blocks-out).
 */
export function toolCallbackToMcpHandler(callback: ToolCallback): McpToolHandler {
  return async (args) => {
    try {
      const raw = await callback.call(JSON.stringify(args ?? {}));
      return {
        content: [{ type: 'text', text: raw }],
      };
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause ?? 'error');
      return {
        isError: true,
        content: [{ type: 'text', text: message }],
      };
    }
  };
}

/**
 * Bundle descriptor + handler for registering on an MCP server.
 */
export function toolCallbackAsMcpTool(callback: ToolCallback): {
  descriptor: McpToolDescriptor;
  handler: McpToolHandler;
  returnDirect: boolean;
} {
  return {
    descriptor: toolCallbackToMcpDescriptor(callback),
    handler: toolCallbackToMcpHandler(callback),
    returnDirect: getToolMetadata(callback).returnDirect,
  };
}
