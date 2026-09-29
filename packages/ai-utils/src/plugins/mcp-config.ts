/**
 * One MCP server entry under Antigravity {@code mcp_config.json} {@code mcpServers}.
 */
export interface AgentPluginMcpServer {
  readonly command?: string;
  readonly serverUrl?: string;
  readonly args?: readonly string[];
  readonly env?: Readonly<Record<string, string>>;
  readonly cwd?: string;
  readonly headers?: Readonly<Record<string, string>>;
  readonly authProviderType?: string;
  readonly oauth?: Readonly<Record<string, unknown>>;
  readonly disabled?: boolean;
  readonly disabledTools?: readonly string[];
  readonly description?: string;
}

/** Parsed Antigravity plugin MCP configuration. */
export interface AgentPluginMcpConfig {
  readonly mcpServers: Readonly<Record<string, AgentPluginMcpServer>>;
}

export type ParseMcpConfigError =
  | 'mcp-config-invalid-root'
  | 'mcp-config-missing-servers'
  | 'mcp-config-servers-invalid'
  | 'mcp-config-server-invalid'
  | 'mcp-config-transport-missing';

export interface ParseMcpConfigResult {
  readonly config?: AgentPluginMcpConfig;
  readonly errors: readonly {
    readonly code: ParseMcpConfigError;
    readonly message: string;
    readonly serverName?: string;
  }[];
}

/**
 * Parse {@code mcp_config.json}. Returns typed config and/or shape errors
 * without throwing.
 */
export function parseMcpConfig(raw: unknown): ParseMcpConfigResult {
  const errors: { code: ParseMcpConfigError; message: string; serverName?: string }[] = [];
  if (raw == null || typeof raw !== 'object' || Array.isArray(raw)) {
    return {
      errors: [
        {
          code: 'mcp-config-invalid-root',
          message: 'mcp_config.json must be a JSON object',
        },
      ],
    };
  }
  const record = raw as Record<string, unknown>;
  if (!Object.hasOwn(record, 'mcpServers')) {
    return {
      errors: [
        {
          code: 'mcp-config-missing-servers',
          message: 'mcp_config.json requires an mcpServers object',
        },
      ],
    };
  }
  if (
    record.mcpServers == null ||
    typeof record.mcpServers !== 'object' ||
    Array.isArray(record.mcpServers)
  ) {
    return {
      errors: [
        {
          code: 'mcp-config-servers-invalid',
          message: 'mcpServers must be an object',
        },
      ],
    };
  }

  const mcpServers: Record<string, AgentPluginMcpServer> = {};
  for (const [serverName, value] of Object.entries(record.mcpServers as Record<string, unknown>)) {
    const parsed = parseMcpServer(serverName, value);
    if (parsed.error) {
      errors.push(parsed.error);
      continue;
    }
    mcpServers[serverName] = parsed.server;
  }

  if (errors.length > 0) {
    return { errors };
  }
  return { config: { mcpServers }, errors: [] };
}

function parseMcpServer(
  serverName: string,
  value: unknown,
):
  | { readonly server: AgentPluginMcpServer; readonly error?: undefined }
  | {
      readonly server?: undefined;
      readonly error: { code: ParseMcpConfigError; message: string; serverName: string };
    } {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    return {
      error: {
        code: 'mcp-config-server-invalid',
        message: `MCP server "${serverName}" must be an object`,
        serverName,
      },
    };
  }
  const entry = value as Record<string, unknown>;
  if (
    Object.hasOwn(entry, 'command') &&
    entry.command != null &&
    typeof entry.command !== 'string'
  ) {
    return {
      error: {
        code: 'mcp-config-server-invalid',
        message: `MCP server "${serverName}" command must be a string`,
        serverName,
      },
    };
  }
  if (
    Object.hasOwn(entry, 'serverUrl') &&
    entry.serverUrl != null &&
    typeof entry.serverUrl !== 'string'
  ) {
    return {
      error: {
        code: 'mcp-config-server-invalid',
        message: `MCP server "${serverName}" serverUrl must be a string`,
        serverName,
      },
    };
  }
  const command = optionalString(entry.command);
  const serverUrl = optionalString(entry.serverUrl);
  if (!command && !serverUrl) {
    return {
      error: {
        code: 'mcp-config-transport-missing',
        message: `MCP server "${serverName}" requires command or serverUrl`,
        serverName,
      },
    };
  }
  if (entry.args != null && !isStringArray(entry.args)) {
    return {
      error: {
        code: 'mcp-config-server-invalid',
        message: `MCP server "${serverName}" args must be a string array`,
        serverName,
      },
    };
  }
  if (entry.env != null && !isStringRecord(entry.env)) {
    return {
      error: {
        code: 'mcp-config-server-invalid',
        message: `MCP server "${serverName}" env must be a string map`,
        serverName,
      },
    };
  }
  if (entry.headers != null && !isStringRecord(entry.headers)) {
    return {
      error: {
        code: 'mcp-config-server-invalid',
        message: `MCP server "${serverName}" headers must be a string map`,
        serverName,
      },
    };
  }
  if (entry.cwd != null && typeof entry.cwd !== 'string') {
    return {
      error: {
        code: 'mcp-config-server-invalid',
        message: `MCP server "${serverName}" cwd must be a string`,
        serverName,
      },
    };
  }
  if (entry.disabled != null && typeof entry.disabled !== 'boolean') {
    return {
      error: {
        code: 'mcp-config-server-invalid',
        message: `MCP server "${serverName}" disabled must be a boolean`,
        serverName,
      },
    };
  }
  if (entry.disabledTools != null && !isStringArray(entry.disabledTools)) {
    return {
      error: {
        code: 'mcp-config-server-invalid',
        message: `MCP server "${serverName}" disabledTools must be a string array`,
        serverName,
      },
    };
  }
  if (entry.description != null && typeof entry.description !== 'string') {
    return {
      error: {
        code: 'mcp-config-server-invalid',
        message: `MCP server "${serverName}" description must be a string`,
        serverName,
      },
    };
  }
  if (entry.oauth != null && (typeof entry.oauth !== 'object' || Array.isArray(entry.oauth))) {
    return {
      error: {
        code: 'mcp-config-server-invalid',
        message: `MCP server "${serverName}" oauth must be an object`,
        serverName,
      },
    };
  }
  if (entry.authProviderType != null && typeof entry.authProviderType !== 'string') {
    return {
      error: {
        code: 'mcp-config-server-invalid',
        message: `MCP server "${serverName}" authProviderType must be a string`,
        serverName,
      },
    };
  }

  return {
    server: {
      command: command || undefined,
      serverUrl: serverUrl || undefined,
      args: entry.args as readonly string[] | undefined,
      env: entry.env as Readonly<Record<string, string>> | undefined,
      cwd: optionalString(entry.cwd),
      headers: entry.headers as Readonly<Record<string, string>> | undefined,
      authProviderType: optionalString(entry.authProviderType),
      oauth: entry.oauth as Readonly<Record<string, unknown>> | undefined,
      disabled: typeof entry.disabled === 'boolean' ? entry.disabled : undefined,
      disabledTools: entry.disabledTools as readonly string[] | undefined,
      description: optionalString(entry.description),
    },
  };
}

function optionalString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : undefined;
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string');
}

function isStringRecord(value: unknown): value is Record<string, string> {
  return (
    value != null &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    Object.values(value as Record<string, unknown>).every((item) => typeof item === 'string')
  );
}
