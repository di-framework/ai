import { Buffer } from 'node:buffer';
import * as fs from 'node:fs';
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import {
  type AiIgnorePolicy,
  type AiIgnoreSuppressionDiagnostic,
  aiIgnoreSuppressionDiagnostic,
  evaluateAiIgnorePath,
  loadAiIgnorePolicy,
} from '../policy/index.ts';
import { assertPathAllowed, expandUserPath } from '../sandbox/paths.ts';
import type {
  AgentSourceDiagnostic,
  AgentSourceDiagnosticCode,
  ResolvedAgentSource,
} from '../sources/resolve-agent-sources.ts';
import { resolveAgentSources } from '../sources/resolve-agent-sources.ts';

export const DEFAULT_AGENT_INSTRUCTIONS_FILENAME = 'AGENTS.md';
export const DEFAULT_AGENT_INSTRUCTIONS_MAX_BYTES = 32 * 1024;

export interface DiscoverAgentInstructionsOptions {
  readonly workspace?: string;
  readonly workingDirectory?: string;
  /** Additional filenames tried after AGENTS.md at each hierarchy level. */
  readonly fallbackFilenames?: readonly string[];
  /** Maximum UTF-8 bytes in the combined content, including separators. */
  readonly maxBytes?: number;
  /** Optional sandbox intersection; does not expand the workspace boundary. */
  readonly allowedDirectories?: readonly string[];
  readonly aiIgnorePolicy?: AiIgnorePolicy;
}

export interface AgentInstructionSource extends ResolvedAgentSource {
  readonly filename: string;
  readonly directory: string;
  readonly content: string;
  readonly bytes: number;
}

export type AgentInstructionsDiagnosticCode =
  | AgentSourceDiagnosticCode
  | 'instructions-invalid-working-directory'
  | 'instructions-outside-allowed-directories'
  | 'instructions-empty'
  | 'instructions-max-bytes-exceeded';

export interface AgentInstructionsLoadDiagnostic {
  readonly code: Exclude<AgentInstructionsDiagnosticCode, AgentSourceDiagnosticCode>;
  readonly severity: 'error' | 'warning';
  readonly path: string;
  readonly origin: 'workspace';
  readonly precedence: number;
  readonly message: string;
}

export type AgentInstructionsDiagnostic =
  | AgentSourceDiagnostic
  | AgentInstructionsLoadDiagnostic
  | AiIgnoreSuppressionDiagnostic;

export interface DiscoverAgentInstructionsResult {
  readonly content: string;
  readonly bytes: number;
  readonly sources: readonly AgentInstructionSource[];
  readonly diagnostics: readonly AgentInstructionsDiagnostic[];
}

/** Discover and load hierarchical repository instructions broad-to-specific. */
export function discoverAgentInstructions(
  options: DiscoverAgentInstructionsOptions = {},
): DiscoverAgentInstructionsResult {
  const workspace = resolve(expandUserPath(options.workspace ?? process.cwd()));
  const workingDirectory = resolveFromWorkspace(
    options.workingDirectory ?? process.cwd(),
    workspace,
  );
  const maxBytes = options.maxBytes ?? DEFAULT_AGENT_INSTRUCTIONS_MAX_BYTES;
  assertValidMaxBytes(maxBytes);
  const filenames = instructionFilenames(options.fallbackFilenames ?? []);
  const workingResolution = resolveAgentSources(
    [{ path: workingDirectory, origin: 'workspace', kind: 'directory' }],
    { workspace },
  );
  if (workingResolution.sources.length === 0) {
    return {
      content: '',
      bytes: 0,
      sources: [],
      diagnostics: workingResolution.diagnostics.map((diagnostic) => ({
        code: 'instructions-invalid-working-directory',
        severity: 'error',
        path: diagnostic.path,
        origin: 'workspace',
        precedence: diagnostic.precedence,
        message: `Working directory is outside or unavailable within the workspace: ${workingDirectory}`,
      })),
    };
  }
  const aiIgnorePolicy = options.aiIgnorePolicy ?? loadAiIgnorePolicy({ workspace });

  const allowedDirectories = options.allowedDirectories?.map((path) =>
    resolveFromWorkspace(path, workspace),
  );
  if (allowedDirectories != null) {
    const access = assertPathAllowed(workingDirectory, allowedDirectories);
    if (!access.ok) {
      return {
        content: '',
        bytes: 0,
        sources: [],
        diagnostics: [
          {
            code: 'instructions-outside-allowed-directories',
            severity: 'error',
            path: workingDirectory,
            origin: 'workspace',
            precedence: 0,
            message: `Working directory is outside the allowed directories: ${workingDirectory}`,
          },
        ],
      };
    }
  }

  const directories = hierarchyDirectories(workspace, workingDirectory);
  const candidates = directories.flatMap((directory) =>
    filenames.map((filename) => ({
      path: join(directory, filename),
      origin: 'workspace' as const,
      kind: 'file' as const,
    })),
  );
  const resolution = resolveAgentSources(candidates, { workspace });
  const suppressionDiagnostics: AiIgnoreSuppressionDiagnostic[] = [];
  const permittedSources = resolution.sources.filter((source) => {
    const evaluation = evaluateAiIgnorePath(aiIgnorePolicy, source.path, { kind: 'file' });
    if (!evaluation.ignored) return true;
    suppressionDiagnostics.push({
      ...aiIgnoreSuppressionDiagnostic(evaluation, 'agent-instructions', 'file'),
      precedence: source.precedence,
    });
    return false;
  });
  const selected = selectOnePerDirectory(permittedSources);
  const diagnostics: AgentInstructionsDiagnostic[] = [
    ...relevantDiagnostics(resolution.diagnostics, selected, filenames.length),
    ...suppressionDiagnostics,
  ];
  const sources: AgentInstructionSource[] = [];
  const parts: string[] = [];
  let bytes = 0;

  for (const source of selected) {
    if (allowedDirectories != null) {
      const access = assertPathAllowed(source.path, allowedDirectories);
      if (!access.ok) {
        diagnostics.push({
          code: 'instructions-outside-allowed-directories',
          severity: 'error',
          path: source.path,
          origin: 'workspace',
          precedence: source.precedence,
          message: `Instruction file is outside the allowed directories: ${source.path}`,
        });
        continue;
      }
    }

    const separatorBytes = sources.length === 0 ? 0 : 2;
    const remaining = maxBytes - bytes - separatorBytes;
    const loaded = readAtMost(source.realPath, remaining);
    if (!loaded.ok) {
      diagnostics.push({
        code: loaded.reason === 'limit' ? 'instructions-max-bytes-exceeded' : 'source-unreadable',
        severity: loaded.reason === 'limit' ? 'warning' : 'error',
        path: source.path,
        origin: 'workspace',
        precedence: source.precedence,
        message:
          loaded.reason === 'limit'
            ? `Instruction file exceeds the remaining ${Math.max(remaining, 0)} byte limit: ${source.path}`
            : `Instruction file became unreadable: ${source.path}`,
      });
      continue;
    }
    const content = loaded.buffer.toString('utf8');
    if (content.trim().length === 0) {
      diagnostics.push({
        code: 'instructions-empty',
        severity: 'warning',
        path: source.path,
        origin: 'workspace',
        precedence: source.precedence,
        message: `Instruction file is empty and was skipped: ${source.path}`,
      });
      continue;
    }

    const sourceBytes = loaded.buffer.byteLength;
    bytes += separatorBytes + sourceBytes;
    parts.push(content);
    sources.push({
      ...source,
      filename: basename(source.path),
      directory: dirname(source.path),
      content,
      bytes: sourceBytes,
    });
  }

  return {
    content: parts.join('\n\n'),
    bytes,
    sources,
    diagnostics: diagnostics.sort(
      (left, right) =>
        (left.precedence ?? Number.MAX_SAFE_INTEGER) -
        (right.precedence ?? Number.MAX_SAFE_INTEGER),
    ),
  };
}

function instructionFilenames(fallbacks: readonly string[]): string[] {
  const out = [DEFAULT_AGENT_INSTRUCTIONS_FILENAME];
  for (const filename of fallbacks) {
    const trimmed = filename.trim();
    if (
      trimmed.length === 0 ||
      trimmed === '.' ||
      trimmed === '..' ||
      trimmed.includes('/') ||
      trimmed.includes('\\')
    ) {
      throw new Error(`Instruction fallback must be a filename: ${filename}`);
    }
    if (!out.includes(trimmed)) out.push(trimmed);
  }
  return out;
}

function assertValidMaxBytes(maxBytes: number): void {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 0) {
    throw new Error('maxBytes must be a non-negative safe integer');
  }
}

function resolveFromWorkspace(path: string, workspace: string): string {
  const expanded = expandUserPath(path);
  return isAbsolute(expanded) ? resolve(expanded) : resolve(workspace, expanded);
}

function hierarchyDirectories(workspace: string, workingDirectory: string): string[] {
  const rel = relative(workspace, workingDirectory);
  if (rel === '') return [workspace];
  const directories = [workspace];
  let current = workspace;
  for (const component of rel.split(sep)) {
    current = join(current, component);
    directories.push(current);
  }
  return directories;
}

function selectOnePerDirectory(
  sources: readonly ResolvedAgentSource[],
): readonly ResolvedAgentSource[] {
  const selected: ResolvedAgentSource[] = [];
  const seen = new Set<string>();
  for (const source of sources) {
    const directory = dirname(source.path);
    if (seen.has(directory)) continue;
    seen.add(directory);
    selected.push(source);
  }
  return selected;
}

function relevantDiagnostics(
  diagnostics: readonly AgentSourceDiagnostic[],
  selected: readonly ResolvedAgentSource[],
  filenamesPerDirectory: number,
): AgentSourceDiagnostic[] {
  const selectedByLevel = new Map(
    selected.map((source) => [
      Math.floor(source.precedence / filenamesPerDirectory),
      source.precedence,
    ]),
  );
  return diagnostics.filter((diagnostic) => {
    const level = Math.floor(diagnostic.precedence / filenamesPerDirectory);
    const selectedPrecedence = selectedByLevel.get(level);
    return selectedPrecedence == null || diagnostic.precedence < selectedPrecedence;
  });
}

type BoundedReadResult =
  | { readonly ok: true; readonly buffer: Buffer }
  | { readonly ok: false; readonly reason: 'limit' | 'unreadable' };

function readAtMost(path: string, limit: number): BoundedReadResult {
  if (limit < 0) return { ok: false, reason: 'limit' };
  let descriptor: number | undefined;
  try {
    descriptor = fs.openSync(path, 'r');
    const buffer = Buffer.alloc(limit + 1);
    let offset = 0;
    while (offset < buffer.length) {
      const read = fs.readSync(descriptor, buffer, offset, buffer.length - offset, null);
      if (read === 0) break;
      offset += read;
    }
    return offset > limit
      ? { ok: false, reason: 'limit' }
      : { ok: true, buffer: buffer.subarray(0, offset) };
  } catch {
    return { ok: false, reason: 'unreadable' };
  } finally {
    if (descriptor != null) fs.closeSync(descriptor);
  }
}
