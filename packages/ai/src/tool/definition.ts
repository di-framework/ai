/**
 * Definition used by the AI model to determine when and how to call a tool.
 * Spring AI: {@code ToolDefinition}.
 */
export interface ToolDefinition {
  /** Unique within the tool set provided to a model. */
  readonly name: string;
  /** What the tool does; used by the model for selection. */
  readonly description: string;
  /** JSON Schema string for tool parameters. */
  readonly inputSchema: string;
}

const EMPTY_OBJECT_SCHEMA = '{"type":"object","properties":{}}';

export function toolDefinition(partial: {
  name: string;
  description?: string;
  inputSchema?: string | Record<string, unknown>;
}): ToolDefinition {
  const name = partial.name.trim();
  if (!name) {
    throw new Error('ToolDefinition.name cannot be empty');
  }
  const description = partial.description?.trim() || humanizeToolName(name);
  const inputSchema = normalizeInputSchema(partial.inputSchema);
  return { name, description, inputSchema };
}

function humanizeToolName(name: string): string {
  return name
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .trim();
}

function normalizeInputSchema(schema?: string | Record<string, unknown>): string {
  if (schema == null) return EMPTY_OBJECT_SCHEMA;
  if (typeof schema === 'string') {
    const trimmed = schema.trim();
    return trimmed.length > 0 ? trimmed : EMPTY_OBJECT_SCHEMA;
  }
  return JSON.stringify(schema);
}

export { EMPTY_OBJECT_SCHEMA as DEFAULT_TOOL_INPUT_SCHEMA };
