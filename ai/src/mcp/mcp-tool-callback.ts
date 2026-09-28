import type { ToolDefinition } from '../tool/definition.ts';
import { ToolExecutionException } from '../tool/execution/tool-execution-exception.ts';
import type { ToolMetadata } from '../tool/metadata.ts';
import { DEFAULT_TOOL_METADATA } from '../tool/metadata.ts';
import type { ToolCallback } from '../tool/tool-callback.ts';
import type { ToolContext } from '../tool/tool-context.ts';
import {
  createToolDefinitionFromMcp,
  defaultToolContextToMcpMetaConverter,
  mcpResultToString,
} from './mcp-tool-utils.ts';
import type {
  McpClientSession,
  McpToolDescriptor,
  ToolContextToMcpMetaConverter,
} from './types.ts';

export interface McpToolCallbackOptions {
  readonly mcpClient: McpClientSession;
  readonly tool: McpToolDescriptor;
  /**
   * Name exposed to the model. Defaults to the original MCP tool name.
   * Provider sets a prefixed name when configured.
   */
  readonly prefixedToolName?: string;
  readonly toolMetadata?: ToolMetadata;
  readonly toolContextToMcpMetaConverter?: ToolContextToMcpMetaConverter;
}

/**
 * Adapts a single MCP tool to {@link ToolCallback}.
 * Spring AI: {@code SyncMcpToolCallback}.
 *
 * {@link call} uses the **original** MCP tool name; {@link toolDefinition}.name
 * may be prefixed for the model.
 */
export class McpToolCallback implements ToolCallback {
  readonly toolDefinition: ToolDefinition;
  readonly toolMetadata: ToolMetadata;
  readonly originalToolName: string;

  private readonly mcpClient: McpClientSession;
  private readonly tool: McpToolDescriptor;
  private readonly metaConverter: ToolContextToMcpMetaConverter;

  constructor(options: McpToolCallbackOptions) {
    if (!options.mcpClient) {
      throw new Error('mcpClient is required');
    }
    if (!options.tool?.name?.trim()) {
      throw new Error('tool with a non-empty name is required');
    }
    this.mcpClient = options.mcpClient;
    this.tool = options.tool;
    this.originalToolName = options.tool.name;
    const name = options.prefixedToolName?.trim() || options.tool.name;
    this.toolDefinition = createToolDefinitionFromMcp(name, options.tool);
    this.toolMetadata = options.toolMetadata ?? DEFAULT_TOOL_METADATA;
    this.metaConverter =
      options.toolContextToMcpMetaConverter ?? defaultToolContextToMcpMetaConverter;
  }

  async call(toolInput: string, toolContext?: ToolContext): Promise<string> {
    let input = toolInput;
    if (!input?.trim()) {
      input = '{}';
    }

    let args: Record<string, unknown>;
    try {
      const parsed = JSON.parse(input) as unknown;
      if (parsed == null || typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error('Tool arguments must be a JSON object');
      }
      args = parsed as Record<string, unknown>;
    } catch (cause) {
      throw new ToolExecutionException(this.toolDefinition, cause, 'Invalid JSON tool arguments');
    }

    try {
      const meta =
        toolContext && !toolContext.isEmpty() ? this.metaConverter(toolContext.context) : undefined;

      const result = await this.mcpClient.callTool({
        name: this.originalToolName,
        arguments: args,
        _meta: meta,
      });

      if (result.isError) {
        throw new ToolExecutionException(
          this.toolDefinition,
          new Error(`Error calling tool: ${mcpResultToString(result)}`),
        );
      }

      return mcpResultToString(result);
    } catch (cause) {
      if (cause instanceof ToolExecutionException) throw cause;
      throw new ToolExecutionException(this.toolDefinition, cause);
    }
  }
}

export function mcpToolCallback(options: McpToolCallbackOptions): McpToolCallback {
  return new McpToolCallback(options);
}
