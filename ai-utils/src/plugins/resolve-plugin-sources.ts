import { homedir } from 'node:os';
import { resolve } from 'node:path';
import type {
  AgentSourceDiagnostic,
  ResolvedAgentSource,
} from '../sources/resolve-agent-sources.ts';
import { resolveAgentSources } from '../sources/resolve-agent-sources.ts';
import { DEFAULT_PLUGIN_DIRECTORY_CANDIDATES } from './load-plugins.ts';
import { resolvePluginPackageDirectories } from './resolve-plugin-packages.ts';

/** Whether explicit plugin roots supplement or replace neutral defaults. */
export type PluginSourceMode = 'merge' | 'replace';

export interface ResolvePluginSourcesOptions {
  readonly workspace?: string;
  readonly userDirectory?: string;
  readonly directories?: readonly string[];
  readonly packages?: readonly string[];
  readonly sourceMode?: PluginSourceMode;
}

export interface ResolvedPluginSources {
  readonly directories: readonly string[];
  readonly sources: readonly ResolvedAgentSource[];
  readonly diagnostics: readonly AgentSourceDiagnostic[];
}

/** Resolve explicit and default plugin roots in runtime precedence order. */
export function resolvePluginSources(
  options: ResolvePluginSourcesOptions = {},
): ResolvedPluginSources {
  const workspace = resolve(options.workspace ?? process.cwd());
  const userDirectory = resolve(options.userDirectory ?? homedir());
  const explicitDirectories = [...(options.directories ?? [])];
  const packageDirectories = options.packages?.length
    ? resolvePluginPackageDirectories(options.packages, workspace)
    : [];
  const allowedDirectories = [...explicitDirectories, ...packageDirectories];
  const defaults =
    options.sourceMode === 'replace'
      ? []
      : [
          { path: DEFAULT_PLUGIN_DIRECTORY_CANDIDATES[0], origin: 'workspace' as const },
          { path: DEFAULT_PLUGIN_DIRECTORY_CANDIDATES[1], origin: 'user' as const },
        ];
  const candidates = [
    ...explicitDirectories.map((path) => ({
      path,
      origin: 'explicit' as const,
      kind: 'directory' as const,
    })),
    ...packageDirectories.map((path) => ({
      path,
      origin: 'package' as const,
      kind: 'directory' as const,
    })),
    ...defaults.map((candidate) => ({ ...candidate, kind: 'directory' as const })),
  ];
  const resolution = resolveAgentSources(candidates, {
    workspace,
    userDirectory,
    allowedDirectories,
  });

  return {
    directories: resolution.sources.map((source) => source.path),
    sources: resolution.sources,
    diagnostics: resolution.diagnostics,
  };
}
