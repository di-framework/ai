import { homedir } from 'node:os';
import { resolve } from 'node:path';
import type {
  AgentSourceDiagnostic,
  ResolvedAgentSource,
} from '../sources/resolve-agent-sources.ts';
import { resolveAgentSources } from '../sources/resolve-agent-sources.ts';
import { DEFAULT_SKILL_DIRECTORY_CANDIDATES } from './load-skills.ts';
import { resolveSkillPackageDirectories } from './resolve-packages.ts';

/** Whether explicit skill roots supplement or replace neutral defaults. */
export type SkillSourceMode = 'merge' | 'replace';

export interface ResolveSkillSourcesOptions {
  readonly workspace?: string;
  readonly userDirectory?: string;
  readonly directories?: readonly string[];
  readonly packages?: readonly string[];
  readonly sourceMode?: SkillSourceMode;
}

export interface ResolvedSkillSources {
  readonly directories: readonly string[];
  readonly sources: readonly ResolvedAgentSource[];
  readonly diagnostics: readonly AgentSourceDiagnostic[];
}

/** Resolve explicit and default skill roots in runtime precedence order. */
export function resolveSkillSources(
  options: ResolveSkillSourcesOptions = {},
): ResolvedSkillSources {
  const workspace = resolve(options.workspace ?? process.cwd());
  const userDirectory = resolve(options.userDirectory ?? homedir());
  const explicitDirectories = [...(options.directories ?? [])];
  const packageDirectories = options.packages?.length
    ? resolveSkillPackageDirectories(options.packages, workspace)
    : [];
  const allowedDirectories = [...explicitDirectories, ...packageDirectories];
  const defaults =
    options.sourceMode === 'replace'
      ? []
      : [
          { path: DEFAULT_SKILL_DIRECTORY_CANDIDATES[0], origin: 'workspace' as const },
          { path: DEFAULT_SKILL_DIRECTORY_CANDIDATES[1], origin: 'user' as const },
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
    // Selecting an explicit root authorizes that root as a skill source.
    allowedDirectories,
  });

  return {
    directories: resolution.sources.map((source) => source.path),
    sources: resolution.sources,
    diagnostics: resolution.diagnostics,
  };
}
