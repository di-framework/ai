import type { ToolCallback } from '../tool/tool-callback.ts';
import { type McpToolHandler, toolCallbackAsMcpTool } from './tool-callback-as-mcp.ts';
import type { McpCallToolResult, McpToolDescriptor } from './types.ts';
export interface McpToolServerOptions {
  readonly name?: string;
  readonly version?: string;
  readonly tools?: readonly ToolCallback[];
}
export interface McpToolServer {
  readonly name: string;
  readonly version: string;
  listTools(): readonly McpToolDescriptor[];
  callTool(name: string, args?: Record<string, unknown>): Promise<McpCallToolResult>;
  addTool(tool: ToolCallback): void;
  removeTool(name: string): boolean;
}
export function createMcpToolServer(options: McpToolServerOptions = {}): McpToolServer {
  const tools = new Map<string, { descriptor: McpToolDescriptor; handler: McpToolHandler }>();
  const addTool = (tool: ToolCallback) => {
    const wrapped = toolCallbackAsMcpTool(tool);
    if (tools.has(wrapped.descriptor.name))
      throw new Error(`Duplicate MCP tool: ${wrapped.descriptor.name}`);
    tools.set(wrapped.descriptor.name, wrapped);
  };
  for (const tool of options.tools ?? []) addTool(tool);
  return {
    name: options.name ?? 'di-framework-ai',
    version: options.version ?? '1.0.0',
    listTools: () => [...tools.values()].map((v) => v.descriptor),
    callTool: async (name, args = {}) => {
      const tool = tools.get(name);
      if (!tool)
        return { isError: true, content: [{ type: 'text', text: `Unknown tool: ${name}` }] };
      return tool.handler(args);
    },
    addTool,
    removeTool: (name) => tools.delete(name),
  };
}
