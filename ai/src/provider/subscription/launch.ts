import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

export const bridgeProviders = ['codex', 'claude', 'agy', 'junie', 'grok'] as const;
export type BridgeProvider = (typeof bridgeProviders)[number];
export const BRIDGE_NAME = 'di_framework_bridge';
export interface StdioConfig {
  command: string;
  args: string[];
  env: Record<string, string>;
}

/** Terella's command/args/env MCP shape, with current provider-specific locations. */
export async function createBridgeLaunch(
  provider: BridgeProvider,
  directory: string,
  server: StdioConfig,
  prompt: string,
  model?: string,
): Promise<string[]> {
  const config = { mcpServers: { [BRIDGE_NAME]: server } };
  async function json(path: string) {
    const file = join(directory, path);
    await mkdir(dirname(file), { recursive: true, mode: 0o700 });
    await writeFile(file, JSON.stringify(config), { mode: 0o600 });
    return file;
  }
  let args: string[];
  switch (provider) {
    case 'codex': {
      // JSON strings/arrays are valid TOML values; env is a TOML inline table.
      const value = `{ command = ${JSON.stringify(server.command)}, args = ${JSON.stringify(server.args)}, env = { ${Object.entries(
        server.env,
      )
        .map(([key, value]) => `${key} = ${JSON.stringify(value)}`)
        .join(', ')} }, required = true, default_tools_approval_mode = "approve" }`;
      args = [
        'exec',
        '--ignore-user-config',
        '--skip-git-repo-check',
        '--sandbox',
        'read-only',
        '-c',
        `mcp_servers.${BRIDGE_NAME}=${value}`,
        prompt,
      ];
      break;
    }
    case 'claude':
      args = [
        '-p',
        prompt,
        '--output-format',
        'text',
        '--tools',
        '',
        '--strict-mcp-config',
        '--mcp-config',
        await json('mcp.json'),
        '--allowedTools',
        `mcp__${BRIDGE_NAME}__*`,
      ];
      break;
    case 'agy':
      await json('.agents/mcp_config.json');
      args = ['-p', prompt, '--output-format', 'text', '--add-dir', directory];
      break;
    case 'junie':
      await json('.junie/mcp/mcp.json');
      args = [
        '--task',
        prompt,
        '--output-format',
        'text',
        '--mcp-default-locations=false',
        '--mcp-location',
        join(directory, '.junie/mcp'),
        '--skip-update-check',
      ];
      break;
    case 'grok':
      await mkdir(join(directory, '.grok'), { recursive: true, mode: 0o700 });
      await writeFile(
        join(directory, '.grok/config.toml'),
        `[mcp_servers.${BRIDGE_NAME}]\ncommand = ${JSON.stringify(server.command)}\nargs = ${JSON.stringify(server.args)}\nenv = { ${Object.entries(
          server.env,
        )
          .map(([key, value]) => `${key} = ${JSON.stringify(value)}`)
          .join(', ')} }\n`,
        { mode: 0o600 },
      );
      args = [
        '-p',
        prompt,
        '--output-format',
        'plain',
        '--tools',
        '',
        // Trust only the freshly generated workspace containing our MCP config.
        '--trust',
        '--no-subagents',
        '--allow',
        `MCPTool(${BRIDGE_NAME}__*)`,
      ];
      break;
  }
  if (model) args.push('--model', model);
  return args;
}
