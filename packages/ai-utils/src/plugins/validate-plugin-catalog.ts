import * as fs from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { nodeErrnoCode } from '../sandbox/fs-error.ts';
import { validateSkillDirectory } from '../skills/validate-skill-catalog.ts';
import type {
  AgentSourceDiagnosticCode,
  AgentSourceOrigin,
  ResolvedAgentSource,
} from '../sources/resolve-agent-sources.ts';
import type { AgentPlugin, AgentPluginRule } from './load-plugins.ts';
import { parseMcpConfig } from './mcp-config.ts';
import { parsePluginManifest } from './parse-plugin-manifest.ts';
import {
  type ResolvedPluginSources,
  type ResolvePluginSourcesOptions,
  resolvePluginSources,
} from './resolve-plugin-sources.ts';
import { validatePluginName } from './validate-plugin.ts';

export type PluginCatalogDiagnosticCode =
  | AgentSourceDiagnosticCode
  | 'plugin-manifest-missing'
  | 'plugin-manifest-invalid'
  | 'plugin-name-invalid'
  | 'plugin-name-directory-mismatch'
  | 'plugin-duplicate'
  | 'plugin-shadowed'
  | 'plugin-mcp-config-invalid'
  | 'plugin-hooks-invalid'
  | 'plugin-rules-unreadable'
  | 'plugin-child-unreadable'
  | 'skill-frontmatter-invalid'
  | 'skill-name-invalid'
  | 'skill-description-invalid'
  | 'skill-name-directory-mismatch'
  | 'skill-entrypoint-missing'
  | 'skill-duplicate'
  | 'skill-shadowed'
  | 'skill-resource-missing'
  | 'skill-resource-unreadable'
  | 'skill-resource-broken-symlink'
  | 'skill-resource-outside-directory';

export interface PluginDiagnosticSource {
  readonly path: string;
  readonly realPath?: string;
  readonly origin?: AgentSourceOrigin;
  readonly precedence?: number;
}

/** A machine-readable validation finding. Messages contain no terminal formatting. */
export interface PluginCatalogDiagnostic {
  readonly code: PluginCatalogDiagnosticCode;
  readonly severity: 'error' | 'warning';
  readonly message: string;
  readonly path: string;
  readonly pluginName?: string;
  readonly source: PluginDiagnosticSource;
  readonly relatedPath?: string;
}

export interface PluginValidationResult {
  readonly valid: boolean;
  readonly plugins: readonly AgentPlugin[];
  readonly diagnostics: readonly PluginCatalogDiagnostic[];
}

export interface ValidatePluginDefinitionOptions {
  readonly path?: string;
  readonly source?: PluginDiagnosticSource;
  readonly matchDirectoryName?: boolean;
}

/** Validate one parsed plugin without spawning MCP servers or running hooks. */
export function validatePluginDefinition(
  plugin: AgentPlugin,
  options: ValidatePluginDefinitionOptions = {},
): PluginValidationResult {
  const path = resolve(options.path ?? plugin.basePath);
  const source = options.source ?? { path };
  const diagnostics: PluginCatalogDiagnostic[] = [];
  const nameError = validatePluginName(plugin.name);
  if (nameError) {
    diagnostics.push(
      diagnostic('plugin-name-invalid', 'error', nameError, path, source, plugin.name),
    );
  }
  if (options.matchDirectoryName === true && plugin.basePath && plugin.basePath !== '.') {
    const directoryName = basename(plugin.basePath);
    if (plugin.name && directoryName !== plugin.name) {
      diagnostics.push(
        diagnostic(
          'plugin-name-directory-mismatch',
          'error',
          `name "${plugin.name}" must match the plugin directory name "${directoryName}"`,
          path,
          source,
          plugin.name,
          plugin.basePath,
        ),
      );
    }
  }
  return validationResult([plugin], diagnostics);
}

/** Validate a single plugin directory, including nested skills and optional configs. */
export function validatePluginDirectory(directory: string): PluginValidationResult {
  const directoryPath = resolve(directory);
  const source: PluginDiagnosticSource = { path: directoryPath };
  const manifestPath = join(directoryPath, 'plugin.json');
  const diagnostics: PluginCatalogDiagnostic[] = [];

  let info: fs.Stats;
  try {
    info = fs.statSync(directoryPath);
  } catch (error) {
    return validationResult(
      [],
      [
        diagnostic(
          nodeErrnoCode(error) === 'ENOENT' && isSymbolicLink(directoryPath)
            ? 'plugin-child-unreadable'
            : 'plugin-manifest-missing',
          'error',
          `Plugin directory is not readable: ${directoryPath}`,
          directoryPath,
          source,
        ),
      ],
    );
  }
  if (!info.isDirectory()) {
    return validationResult(
      [],
      [
        diagnostic(
          'plugin-manifest-missing',
          'error',
          'Plugin path must be a directory containing plugin.json',
          manifestPath,
          source,
        ),
      ],
    );
  }

  let manifestText: string;
  try {
    manifestText = fs.readFileSync(manifestPath, 'utf8');
  } catch (error) {
    const broken = nodeErrnoCode(error) === 'ENOENT' && isSymbolicLink(manifestPath);
    return validationResult(
      [],
      [
        diagnostic(
          broken ? 'plugin-child-unreadable' : 'plugin-manifest-missing',
          'error',
          broken
            ? `Plugin manifest is a broken symlink: ${manifestPath}`
            : `Plugin directory is missing a readable plugin.json: ${manifestPath}`,
          manifestPath,
          source,
        ),
      ],
    );
  }

  let manifestRaw: unknown;
  try {
    manifestRaw = JSON.parse(manifestText) as unknown;
  } catch {
    return validationResult(
      [],
      [
        diagnostic(
          'plugin-manifest-invalid',
          'error',
          `plugin.json is not valid JSON: ${manifestPath}`,
          manifestPath,
          source,
        ),
      ],
    );
  }

  let name = basename(directoryPath);
  let description: string | undefined;
  let version: string | undefined;
  try {
    const manifest = parsePluginManifest(manifestRaw, { fallbackName: name });
    name = manifest.name;
    description = manifest.description;
    version = manifest.version;
  } catch (error) {
    diagnostics.push(
      diagnostic(
        'plugin-manifest-invalid',
        'error',
        error instanceof Error ? error.message : 'plugin.json is invalid',
        manifestPath,
        source,
      ),
    );
  }

  const mcp = readOptionalMcpConfig(directoryPath, source, name);
  diagnostics.push(...mcp.diagnostics);
  const hooks = readOptionalHooks(directoryPath, source, name);
  diagnostics.push(...hooks.diagnostics);
  const rules = readRules(directoryPath, source, name);
  diagnostics.push(...rules.diagnostics);

  const skillsDirectoryPath = join(directoryPath, 'skills');
  const hasSkills = isExistingDirectory(skillsDirectoryPath);
  if (hasSkills) {
    for (const skillDir of listImmediateDirectories(skillsDirectoryPath)) {
      const skillResult = validateSkillDirectory(skillDir);
      for (const item of skillResult.diagnostics) {
        diagnostics.push({
          code: item.code as PluginCatalogDiagnosticCode,
          severity: item.severity,
          message: item.message,
          path: item.path,
          pluginName: name,
          source,
          relatedPath: item.relatedPath,
        });
      }
    }
  }

  if (
    diagnostics.some((item) => item.severity === 'error' && item.code.startsWith('plugin-manifest'))
  ) {
    return validationResult([], diagnostics);
  }

  const plugin: AgentPlugin = {
    name,
    description,
    version,
    basePath: directoryPath,
    manifestPath,
    skillsDirectory: hasSkills ? skillsDirectoryPath : undefined,
    rules: rules.rules,
    mcpConfig: mcp.config,
    hooks: hooks.hooks,
  };
  diagnostics.push(...validatePluginDefinition(plugin, { path: manifestPath, source }).diagnostics);
  return validationResult([plugin], diagnostics);
}

/** Validate every discovered plugin under one catalog directory. */
export function validatePluginsDirectory(directory: string): PluginValidationResult {
  return validatePluginCatalog({ directories: [directory], sourceMode: 'replace' });
}

/** Validate an already-resolved catalog in the exact source precedence supplied. */
export function validateResolvedPluginCatalog(
  resolvedSources: ResolvedPluginSources,
): PluginValidationResult {
  const diagnostics = resolvedSources.diagnostics.map((item) =>
    diagnostic(
      item.code,
      item.severity,
      item.message,
      item.path,
      {
        path: item.path,
        origin: item.origin,
        precedence: item.precedence,
      },
      undefined,
      item.duplicateOf,
    ),
  );
  const plugins: AgentPlugin[] = [];
  const definitions = new Map<string, { plugin: AgentPlugin; source: ResolvedAgentSource }>();

  for (const source of resolvedSources.sources) {
    const sourceContext: PluginDiagnosticSource = source;
    let children: string[];
    try {
      children = listPluginChildren(source.path);
    } catch {
      diagnostics.push(
        diagnostic(
          'source-unreadable',
          'error',
          `Source is unreadable: ${source.path}`,
          source.path,
          sourceContext,
        ),
      );
      continue;
    }
    for (const child of children) {
      const result = validatePluginDirectory(child);
      const sourceDiagnostics = result.diagnostics.map((item) => ({
        ...item,
        source: sourceContext,
      }));
      diagnostics.push(...sourceDiagnostics);
      const plugin = result.plugins[0];
      if (plugin == null) continue;
      const kept = definitions.get(plugin.name);
      if (kept != null) {
        const shadowed = kept.source.precedence !== source.precedence;
        diagnostics.push(
          diagnostic(
            shadowed ? 'plugin-shadowed' : 'plugin-duplicate',
            'warning',
            shadowed
              ? `Plugin "${plugin.name}" is shadowed by higher-precedence definition ${kept.plugin.basePath}`
              : `Plugin "${plugin.name}" duplicates definition ${kept.plugin.basePath}`,
            plugin.basePath,
            sourceContext,
            plugin.name,
            kept.plugin.basePath,
          ),
        );
        continue;
      }
      definitions.set(plugin.name, { plugin, source });
      plugins.push(plugin);
    }
  }
  return validationResult(plugins, diagnostics);
}

/** Resolve and validate a catalog using the same roots and precedence as runtime discovery. */
export function validatePluginCatalog(
  options: ResolvePluginSourcesOptions = {},
): PluginValidationResult {
  return validateResolvedPluginCatalog(resolvePluginSources(options));
}

function listPluginChildren(root: string): string[] {
  // Single-plugin roots publish {@code plugin.json} at the catalog path itself.
  if (isExistingFile(join(root, 'plugin.json'))) {
    return [root];
  }
  const out: string[] = [];
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    if (!entry.isDirectory() && !entry.isSymbolicLink()) continue;
    const child = join(root, entry.name);
    try {
      if (!fs.statSync(child).isDirectory()) continue;
    } catch {
      continue;
    }
    out.push(child);
  }
  return out.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

function listImmediateDirectories(root: string): string[] {
  const out: string[] = [];
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    if (!entry.isDirectory() && !entry.isSymbolicLink()) continue;
    const child = join(root, entry.name);
    try {
      if (!fs.statSync(child).isDirectory()) continue;
    } catch {
      continue;
    }
    out.push(child);
  }
  return out;
}

function readOptionalMcpConfig(
  directoryPath: string,
  source: PluginDiagnosticSource,
  pluginName: string,
): {
  readonly config?: AgentPlugin['mcpConfig'];
  readonly diagnostics: PluginCatalogDiagnostic[];
} {
  const path = join(directoryPath, 'mcp_config.json');
  let text: string;
  try {
    text = fs.readFileSync(path, 'utf8');
  } catch {
    return { diagnostics: [] };
  }
  let raw: unknown;
  try {
    raw = JSON.parse(text) as unknown;
  } catch {
    return {
      diagnostics: [
        diagnostic(
          'plugin-mcp-config-invalid',
          'error',
          `mcp_config.json is not valid JSON: ${path}`,
          path,
          source,
          pluginName,
        ),
      ],
    };
  }
  const parsed = parseMcpConfig(raw);
  if (parsed.errors.length > 0) {
    return {
      diagnostics: parsed.errors.map((error) =>
        diagnostic('plugin-mcp-config-invalid', 'error', error.message, path, source, pluginName),
      ),
    };
  }
  return { config: parsed.config, diagnostics: [] };
}

function readOptionalHooks(
  directoryPath: string,
  source: PluginDiagnosticSource,
  pluginName: string,
): {
  readonly hooks?: Readonly<Record<string, unknown>>;
  readonly diagnostics: PluginCatalogDiagnostic[];
} {
  const path = join(directoryPath, 'hooks.json');
  let text: string;
  try {
    text = fs.readFileSync(path, 'utf8');
  } catch {
    return { diagnostics: [] };
  }
  let raw: unknown;
  try {
    raw = JSON.parse(text) as unknown;
  } catch {
    return {
      diagnostics: [
        diagnostic(
          'plugin-hooks-invalid',
          'error',
          `hooks.json is not valid JSON: ${path}`,
          path,
          source,
          pluginName,
        ),
      ],
    };
  }
  if (raw == null || typeof raw !== 'object' || Array.isArray(raw)) {
    return {
      diagnostics: [
        diagnostic(
          'plugin-hooks-invalid',
          'error',
          `hooks.json must be a JSON object: ${path}`,
          path,
          source,
          pluginName,
        ),
      ],
    };
  }
  return { hooks: raw as Readonly<Record<string, unknown>>, diagnostics: [] };
}

function readRules(
  directoryPath: string,
  source: PluginDiagnosticSource,
  pluginName: string,
): {
  readonly rules: AgentPluginRule[];
  readonly diagnostics: PluginCatalogDiagnostic[];
} {
  const rulesDir = join(directoryPath, 'rules');
  if (!isExistingDirectory(rulesDir)) return { rules: [], diagnostics: [] };
  const rules: AgentPluginRule[] = [];
  const diagnostics: PluginCatalogDiagnostic[] = [];
  for (const entry of fs.readdirSync(rulesDir, { withFileTypes: true })) {
    if (!entry.isFile() && !entry.isSymbolicLink()) continue;
    if (!entry.name.endsWith('.md')) continue;
    const path = join(rulesDir, entry.name);
    try {
      const content = fs.readFileSync(path, 'utf8');
      rules.push({
        name: basename(entry.name, '.md'),
        path,
        content,
      });
    } catch {
      diagnostics.push(
        diagnostic(
          'plugin-rules-unreadable',
          'error',
          `Rule file is unreadable: ${path}`,
          path,
          source,
          pluginName,
        ),
      );
    }
  }
  return {
    rules: rules.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0)),
    diagnostics,
  };
}

function isExistingDirectory(dir: string): boolean {
  try {
    return fs.statSync(dir).isDirectory();
  } catch {
    return false;
  }
}

function isExistingFile(path: string): boolean {
  try {
    return fs.statSync(path).isFile();
  } catch {
    return false;
  }
}

function isSymbolicLink(path: string): boolean {
  try {
    return fs.lstatSync(path).isSymbolicLink();
  } catch {
    return false;
  }
}

function diagnostic(
  code: PluginCatalogDiagnosticCode,
  severity: 'error' | 'warning',
  message: string,
  path: string,
  source: PluginDiagnosticSource,
  pluginName?: string,
  relatedPath?: string,
): PluginCatalogDiagnostic {
  return {
    code,
    severity,
    message,
    path,
    pluginName,
    source,
    relatedPath,
  };
}

function validationResult(
  plugins: readonly AgentPlugin[],
  diagnostics: readonly PluginCatalogDiagnostic[],
): PluginValidationResult {
  return {
    valid: diagnostics.every((item) => item.severity !== 'error'),
    plugins,
    diagnostics,
  };
}
