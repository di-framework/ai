import { readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { nodeErrnoCode } from '../sandbox/fs-error.ts';
import { expandUserPath } from '../sandbox/paths.ts';
import { type AgentPluginMcpConfig, parseMcpConfig } from './mcp-config.ts';
import { parsePluginManifest } from './parse-plugin-manifest.ts';

/** Vendor-neutral workspace and user locations used by merge discovery. */
export const DEFAULT_PLUGIN_DIRECTORY_CANDIDATES = [
  '.agents/plugins',
  '~/.agents/plugins',
] as const;

/** One markdown rule file discovered under a plugin {@code rules/} directory. */
export interface AgentPluginRule {
  readonly name: string;
  readonly path: string;
  readonly content: string;
}

/** A loaded Antigravity-shaped plugin bundle. */
export interface AgentPlugin {
  readonly name: string;
  readonly description?: string;
  readonly version?: string;
  readonly basePath: string;
  readonly manifestPath: string;
  readonly skillsDirectory?: string;
  readonly rules: readonly AgentPluginRule[];
  readonly mcpConfig?: AgentPluginMcpConfig;
  readonly hooks?: Readonly<Record<string, unknown>>;
}

/**
 * Candidates that exist as directories. Missing paths are skipped so default
 * {@code ~/.agents/plugins} does not fail closed on a fresh machine.
 */
export function existingPluginDirectories(
  candidates: readonly string[] = DEFAULT_PLUGIN_DIRECTORY_CANDIDATES,
): string[] {
  const out: string[] = [];
  for (const candidate of candidates) {
    const resolved = resolve(expandUserPath(candidate));
    try {
      if (statSync(resolved).isDirectory()) {
        out.push(resolved);
      }
    } catch {
      // Missing or unreadable candidates are skipped.
    }
  }
  return out.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

/**
 * Load one plugin directory that contains {@code plugin.json}.
 */
export function loadPluginDirectory(pluginDirectory: string): AgentPlugin {
  const basePath = resolve(pluginDirectory);
  let isDirectory = false;
  try {
    isDirectory = statSync(basePath).isDirectory();
  } catch {
    throw new Error(`Plugin directory does not exist: ${pluginDirectory}`);
  }
  if (!isDirectory) {
    throw new Error(`Path is not a directory: ${pluginDirectory}`);
  }

  const manifestPath = join(basePath, 'plugin.json');
  let manifestRaw: unknown;
  try {
    manifestRaw = JSON.parse(readFileSync(manifestPath, 'utf8')) as unknown;
  } catch (error) {
    if (nodeErrnoCode(error) === 'ENOENT') {
      throw new Error(`Plugin is missing plugin.json: ${manifestPath}`);
    }
    throw new Error(`Unable to read plugin.json: ${manifestPath}`);
  }

  const manifest = parsePluginManifest(manifestRaw, {
    fallbackName: basename(basePath),
  });

  const skillsDirectoryPath = join(basePath, 'skills');
  const skillsDirectory = isExistingDirectory(skillsDirectoryPath)
    ? skillsDirectoryPath
    : undefined;

  return {
    name: manifest.name,
    description: manifest.description,
    version: manifest.version,
    basePath,
    manifestPath,
    skillsDirectory,
    rules: loadRules(basePath),
    mcpConfig: loadMcpConfig(basePath),
    hooks: loadHooks(basePath),
  };
}

/**
 * Load every immediate child directory under {@code rootDirectory} that
 * contains a readable {@code plugin.json}.
 */
export function loadPluginsDirectory(rootDirectory: string): AgentPlugin[] {
  const rootPath = resolve(rootDirectory);
  let isDirectory = false;
  try {
    isDirectory = statSync(rootPath).isDirectory();
  } catch {
    throw new Error(`Root directory does not exist: ${rootDirectory}`);
  }
  if (!isDirectory) {
    throw new Error(`Path is not a directory: ${rootDirectory}`);
  }

  // A package root that is itself a plugin (has plugin.json) is a single-plugin root.
  if (isExistingFile(join(rootPath, 'plugin.json'))) {
    return [loadPluginDirectory(rootPath)];
  }

  const plugins: AgentPlugin[] = [];
  for (const entry of readdirSync(rootPath, { withFileTypes: true })) {
    if (!entry.isDirectory() && !entry.isSymbolicLink()) continue;
    const child = join(rootPath, entry.name);
    try {
      if (!statSync(child).isDirectory()) continue;
    } catch {
      continue;
    }
    const manifestPath = join(child, 'plugin.json');
    try {
      statSync(manifestPath);
    } catch {
      continue;
    }
    plugins.push(loadPluginDirectory(child));
  }
  return plugins.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

/**
 * Load plugins from each root directory.
 */
export function loadPluginsDirectories(rootDirectories: readonly string[]): AgentPlugin[] {
  const plugins: AgentPlugin[] = [];
  for (const root of rootDirectories) {
    plugins.push(...loadPluginsDirectory(root));
  }
  return plugins;
}

function loadRules(basePath: string): AgentPluginRule[] {
  const rulesDir = join(basePath, 'rules');
  if (!isExistingDirectory(rulesDir)) return [];
  const rules: AgentPluginRule[] = [];
  for (const entry of readdirSync(rulesDir, { withFileTypes: true })) {
    if (!entry.isFile() && !entry.isSymbolicLink()) continue;
    if (!entry.name.endsWith('.md')) continue;
    const path = join(rulesDir, entry.name);
    try {
      const content = readFileSync(path, 'utf8');
      rules.push({
        name: basename(entry.name, '.md'),
        path,
        content,
      });
    } catch {
      // Skip unreadable rule files during load; catalog validation reports them.
    }
  }
  return rules.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

function loadMcpConfig(basePath: string): AgentPluginMcpConfig | undefined {
  const path = join(basePath, 'mcp_config.json');
  let rawText: string;
  try {
    rawText = readFileSync(path, 'utf8');
  } catch {
    return undefined;
  }
  let raw: unknown;
  try {
    raw = JSON.parse(rawText) as unknown;
  } catch {
    throw new Error(`Unable to parse mcp_config.json: ${path}`);
  }
  const parsed = parseMcpConfig(raw);
  if (parsed.errors.length > 0) {
    throw new Error(
      `Invalid mcp_config.json (${path}): ${parsed.errors.map((error) => error.message).join('; ')}`,
    );
  }
  return parsed.config;
}

function loadHooks(basePath: string): Readonly<Record<string, unknown>> | undefined {
  const path = join(basePath, 'hooks.json');
  let rawText: string;
  try {
    rawText = readFileSync(path, 'utf8');
  } catch {
    return undefined;
  }
  let raw: unknown;
  try {
    raw = JSON.parse(rawText) as unknown;
  } catch {
    throw new Error(`Unable to parse hooks.json: ${path}`);
  }
  if (raw == null || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error(`hooks.json must be a JSON object: ${path}`);
  }
  return raw as Readonly<Record<string, unknown>>;
}

function isExistingDirectory(dir: string): boolean {
  try {
    return statSync(dir).isDirectory();
  } catch {
    return false;
  }
}

function isExistingFile(path: string): boolean {
  try {
    return statSync(path).isFile();
  } catch {
    return false;
  }
}
