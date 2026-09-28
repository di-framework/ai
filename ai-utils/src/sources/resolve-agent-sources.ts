import * as fs from 'node:fs';
import { constants } from 'node:fs';
import { homedir } from 'node:os';
import { isAbsolute, relative, resolve, sep } from 'node:path';
import { nodeErrnoCode } from '../sandbox/fs-error.ts';

/** Describes why an operational source was considered. */
export type AgentSourceOrigin =
  | 'explicit'
  | 'workspace'
  | 'user'
  | 'package'
  | 'fallback'
  | 'vendor'
  | 'migration';

/** Filesystem shape required by a source consumer. */
export type AgentSourceKind = 'file' | 'directory' | 'any';

/** One source candidate. Candidates are evaluated in array order. */
export interface AgentSourceCandidate {
  readonly path: string;
  readonly origin: AgentSourceOrigin;
  readonly kind?: AgentSourceKind;
}

export interface ResolveAgentSourcesOptions {
  /** Root used for relative paths and the boundary for workspace sources. */
  readonly workspace: string;
  /** Additional roots trusted for explicit, package, fallback, vendor, or migration sources. */
  readonly allowedDirectories?: readonly string[];
  /** Home root used for `~` expansion and the boundary for user sources. */
  readonly userDirectory?: string;
}

/** A readable, boundary-checked source. Lower precedence values win. */
export interface ResolvedAgentSource {
  readonly path: string;
  readonly realPath: string;
  readonly origin: AgentSourceOrigin;
  readonly precedence: number;
  readonly kind: Exclude<AgentSourceKind, 'any'>;
}

export type AgentSourceDiagnosticCode =
  | 'source-missing'
  | 'source-unreadable'
  | 'source-kind-mismatch'
  | 'source-broken-symlink'
  | 'source-outside-boundary'
  | 'source-duplicate';

export interface AgentSourceDiagnostic {
  readonly code: AgentSourceDiagnosticCode;
  readonly severity: 'error' | 'warning';
  readonly path: string;
  readonly origin: AgentSourceOrigin;
  readonly precedence: number;
  readonly message: string;
  readonly duplicateOf?: string;
}

export interface ResolveAgentSourcesResult {
  readonly sources: readonly ResolvedAgentSource[];
  readonly diagnostics: readonly AgentSourceDiagnostic[];
}

/**
 * Resolve ordered operational sources without loading their contents.
 *
 * Relative candidates are resolved from `workspace`; `~` uses `userDirectory`
 * (or the process home). Duplicate real paths are ignored after their first
 * occurrence. Every rejected or shadowed candidate produces one stable,
 * source-aware diagnostic.
 */
export function resolveAgentSources(
  candidates: readonly AgentSourceCandidate[],
  options: ResolveAgentSourcesOptions,
): ResolveAgentSourcesResult {
  const userDirectory = resolve(options.userDirectory ?? homedir());
  const workspace = resolveWithUser(options.workspace, process.cwd(), userDirectory);
  const generalRoots = uniqueRoots([
    workspace,
    ...(options.allowedDirectories ?? []).map((root) =>
      resolveWithUser(root, workspace, userDirectory),
    ),
  ]);
  const workspaceRoots = uniqueRoots([workspace]);
  const userRoots = uniqueRoots([userDirectory]);
  const sources: ResolvedAgentSource[] = [];
  const diagnostics: AgentSourceDiagnostic[] = [];
  const seen = new Map<string, string>();

  for (const [precedence, candidate] of candidates.entries()) {
    const path = resolveWithUser(candidate.path, workspace, userDirectory);
    const roots =
      candidate.origin === 'workspace'
        ? workspaceRoots
        : candidate.origin === 'user'
          ? userRoots
          : generalRoots;
    const base = { path, origin: candidate.origin, precedence } as const;

    if (!roots.some((root) => isContained(path, root.path))) {
      diagnostics.push({
        ...base,
        code: 'source-outside-boundary',
        severity: 'error',
        message: `Source is outside the ${boundaryName(candidate.origin)} boundary: ${path}`,
      });
      continue;
    }

    let info: fs.Stats;
    try {
      info = fs.statSync(path);
    } catch (error) {
      const code = nodeErrnoCode(error);
      const brokenSymlink = code === 'ENOENT' && isSymbolicLink(path);
      diagnostics.push({
        ...base,
        code: brokenSymlink
          ? 'source-broken-symlink'
          : code === 'ENOENT'
            ? 'source-missing'
            : 'source-unreadable',
        severity: 'error',
        message: brokenSymlink
          ? `Source is a broken symlink: ${path}`
          : code === 'ENOENT'
            ? `Source does not exist: ${path}`
            : `Source is unreadable: ${path}`,
      });
      continue;
    }

    let realPath: string;
    try {
      realPath = fs.realpathSync(path);
    } catch {
      diagnostics.push({
        ...base,
        code: 'source-unreadable',
        severity: 'error',
        message: `Source is unreadable: ${path}`,
      });
      continue;
    }

    if (!roots.some((root) => root.realPath != null && isContained(realPath, root.realPath))) {
      diagnostics.push({
        ...base,
        code: 'source-outside-boundary',
        severity: 'error',
        message: `Source resolves outside the ${boundaryName(candidate.origin)} boundary: ${path}`,
      });
      continue;
    }

    try {
      fs.accessSync(realPath, constants.R_OK);
    } catch {
      diagnostics.push({
        ...base,
        code: 'source-unreadable',
        severity: 'error',
        message: `Source is unreadable: ${path}`,
      });
      continue;
    }

    const kind = info.isDirectory() ? 'directory' : info.isFile() ? 'file' : undefined;
    if (
      kind == null ||
      (candidate.kind != null && candidate.kind !== 'any' && candidate.kind !== kind)
    ) {
      const expected = candidate.kind ?? 'file or directory';
      diagnostics.push({
        ...base,
        code: 'source-kind-mismatch',
        severity: 'error',
        message: `Source must be ${article(expected)} ${expected}: ${path}`,
      });
      continue;
    }

    const duplicateOf = seen.get(realPath);
    if (duplicateOf != null) {
      diagnostics.push({
        ...base,
        code: 'source-duplicate',
        severity: 'warning',
        message: `Source duplicates higher-precedence source ${duplicateOf}: ${path}`,
        duplicateOf,
      });
      continue;
    }

    seen.set(realPath, path);
    sources.push({ ...base, realPath, kind });
  }

  return { sources, diagnostics };
}

interface ResolvedRoot {
  readonly path: string;
  readonly realPath?: string;
}

function uniqueRoots(paths: readonly string[]): ResolvedRoot[] {
  const roots: ResolvedRoot[] = [];
  const seen = new Set<string>();
  for (const path of paths) {
    let realPath: string | undefined;
    try {
      realPath = fs.realpathSync(path);
    } catch {
      // A missing allowed root can still classify a missing candidate, but it
      // can never authorize an existing source without a verified real path.
    }
    const key = realPath ?? path;
    if (seen.has(key)) continue;
    seen.add(key);
    roots.push({ path, realPath });
  }
  return roots;
}

function resolveWithUser(input: string, base: string, userDirectory: string): string {
  const trimmed = input.trim();
  const expanded =
    trimmed === '~'
      ? userDirectory
      : trimmed.startsWith('~/') || trimmed.startsWith(`~${sep}`)
        ? resolve(userDirectory, trimmed.slice(2))
        : trimmed;
  return isAbsolute(expanded) ? resolve(expanded) : resolve(base, expanded);
}

function isContained(target: string, root: string): boolean {
  if (target === root) return true;
  const rel = relative(root, target);
  return rel !== '' && rel !== '..' && !rel.startsWith(`..${sep}`) && !isAbsolute(rel);
}

function isSymbolicLink(path: string): boolean {
  try {
    return fs.lstatSync(path).isSymbolicLink();
  } catch {
    return false;
  }
}

function boundaryName(origin: AgentSourceOrigin): string {
  if (origin === 'workspace') return 'workspace';
  if (origin === 'user') return 'user directory';
  return 'allowed-directory';
}

function article(value: string): 'a' | 'an' {
  return /^[aeiou]/i.test(value) ? 'an' : 'a';
}
