import * as fs from 'node:fs';
import { isAbsolute, relative, resolve, sep } from 'node:path';
import { assertPathAllowed, type PathAccessResult } from '../sandbox/index.ts';

export const AIIGNORE_FILE = '.aiignore';

export interface AiIgnorePolicySource {
  /** Lexical root-workspace policy path. */
  readonly path: string;
  /** Canonical policy path when it exists. */
  readonly realPath?: string;
  readonly workspace: string;
  readonly realWorkspace: string;
  readonly exists: boolean;
}

export interface AiIgnoreRule {
  readonly source: AiIgnorePolicySource;
  readonly line: number;
  readonly original: string;
  readonly pattern: string;
  readonly negated: boolean;
  readonly directoryOnly: boolean;
  readonly rootRelative: boolean;
}

export interface AiIgnorePolicy {
  readonly source: AiIgnorePolicySource;
  readonly rules: readonly AiIgnoreRule[];
}

export interface CompileAiIgnorePolicyOptions {
  readonly workspace: string;
  /** Provenance label for explicitly supplied text. Loading remains root-only. */
  readonly sourcePath?: string;
  readonly sourceRealPath?: string;
  readonly sourceExists?: boolean;
}

export interface LoadAiIgnorePolicyOptions {
  readonly workspace: string;
}

export type AiIgnoreDecision =
  | 'sandbox-denied'
  | 'policy-file'
  | 'ignored'
  | 'included'
  | 'unmatched';

export interface EvaluateAiIgnorePathOptions {
  /** Required only when evaluating a path that does not yet exist. */
  readonly kind?: 'file' | 'directory';
}

export interface AiIgnoreEvaluation {
  readonly path: string;
  readonly relativePath?: string;
  readonly ignored: boolean;
  readonly decision: AiIgnoreDecision;
  readonly source: AiIgnorePolicySource;
  readonly rule?: AiIgnoreRule;
  /** The sandbox decision is made before any policy rule is considered. */
  readonly pathAccess: PathAccessResult;
}

export type AiIgnoreDiscoverySurface =
  | 'recursive-walk'
  | 'glob'
  | 'grep'
  | 'skill-discovery'
  | 'agent-instructions';

/** Content-free record emitted when discovery suppresses an operational path. */
export interface AiIgnoreSuppressionDiagnostic {
  readonly code: 'aiignore-suppressed';
  readonly severity: 'warning';
  readonly path: string;
  readonly kind: 'file' | 'directory';
  readonly surface: AiIgnoreDiscoverySurface;
  readonly policyPath: string;
  readonly policyLine?: number;
  readonly precedence?: number;
  readonly message: string;
}

export type AiIgnorePolicyErrorCode =
  | 'WORKSPACE_UNAVAILABLE'
  | 'POLICY_OUTSIDE_WORKSPACE'
  | 'POLICY_UNREADABLE';

export class AiIgnorePolicyError extends Error {
  override readonly name = 'AiIgnorePolicyError';

  constructor(
    readonly code: AiIgnorePolicyErrorCode,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
  }
}

const matchers = new WeakMap<AiIgnoreRule, RegExp>();

/** Compile explicit `.gitignore`-style text without reading process or CLI state. */
export function compileAiIgnorePolicy(
  content: string,
  options: CompileAiIgnorePolicyOptions,
): AiIgnorePolicy {
  const source = policySource(options);
  const rules: AiIgnoreRule[] = [];
  for (const [offset, input] of content.split(/\r?\n/).entries()) {
    const original = input.endsWith('\r') ? input.slice(0, -1) : input;
    let pattern = trimUnescapedTrailingSpaces(original);
    if (pattern === '' || isUnescapedPrefix(pattern, '#')) continue;

    const negated = isUnescapedPrefix(pattern, '!');
    if (negated) pattern = pattern.slice(1);
    if (pattern === '') continue;

    const directoryOnly = endsWithUnescapedSlash(pattern);
    if (directoryOnly) pattern = pattern.slice(0, -1);
    const rootRelative = pattern.startsWith('/');
    if (rootRelative) pattern = pattern.slice(1);
    if (pattern === '') continue;

    const normalizedPattern = normalizePattern(pattern);
    const rule: AiIgnoreRule = {
      source,
      line: offset + 1,
      original,
      pattern: normalizedPattern,
      negated,
      directoryOnly,
      rootRelative,
    };
    matchers.set(rule, compileMatcher(normalizedPattern, rootRelative));
    rules.push(rule);
  }
  return { source, rules };
}

/** Load only `<workspace>/.aiignore`; nested policy discovery is intentionally absent. */
export function loadAiIgnorePolicy(options: LoadAiIgnorePolicyOptions): AiIgnorePolicy {
  const workspace = resolve(options.workspace);
  let realWorkspace: string;
  try {
    realWorkspace = fs.realpathSync(workspace);
  } catch (error) {
    throw new AiIgnorePolicyError(
      'WORKSPACE_UNAVAILABLE',
      `Workspace does not exist or cannot be resolved: ${workspace}`,
      { cause: error },
    );
  }
  const sourcePath = resolve(workspace, AIIGNORE_FILE);
  const access = assertPathAllowed(sourcePath, [workspace]);
  if (!access.ok) {
    throw new AiIgnorePolicyError('POLICY_OUTSIDE_WORKSPACE', access.error);
  }

  let sourceRealPath: string;
  let content: string;
  try {
    sourceRealPath = fs.realpathSync(sourcePath);
    content = fs.readFileSync(sourceRealPath, 'utf8');
  } catch (error) {
    if (isErrno(error, 'ENOENT')) {
      return compileAiIgnorePolicy('', {
        workspace,
        sourcePath,
        sourceExists: false,
      });
    }
    throw new AiIgnorePolicyError('POLICY_UNREADABLE', `Policy is unreadable: ${sourcePath}`, {
      cause: error,
    });
  }

  const canonicalAccess = assertPathAllowed(sourceRealPath, [realWorkspace]);
  if (!canonicalAccess.ok) {
    throw new AiIgnorePolicyError('POLICY_OUTSIDE_WORKSPACE', canonicalAccess.error);
  }
  return compileAiIgnorePolicy(content, {
    workspace,
    sourcePath,
    sourceRealPath,
    sourceExists: true,
  });
}

/** Evaluate one path after the workspace sandbox has accepted its lexical and real boundary. */
export function evaluateAiIgnorePath(
  policy: AiIgnorePolicy,
  input: string,
  options: EvaluateAiIgnorePathOptions = {},
): AiIgnoreEvaluation {
  const candidate = candidatePath(policy.source.workspace, input);
  const path = resolveCandidate(policy.source.workspace, input);
  const pathAccess = assertPathAllowed(candidate, [policy.source.workspace]);
  if (!pathAccess.ok) {
    return {
      path,
      ignored: false,
      decision: 'sandbox-denied',
      source: policy.source,
      pathAccess,
    };
  }

  const relativePath = normalizeRelativePath(relative(policy.source.workspace, path));
  const realPath = existingRealPath(path);
  if (
    path === policy.source.path ||
    (realPath != null && policy.source.realPath != null && realPath === policy.source.realPath)
  ) {
    return {
      path,
      relativePath,
      ignored: false,
      decision: 'policy-file',
      source: policy.source,
      pathAccess,
    };
  }

  const kind = options.kind ?? existingKind(path);
  let matched: AiIgnoreRule | undefined;
  let ignored = false;
  for (const rule of policy.rules) {
    if (!ruleMatches(rule, relativePath, kind)) continue;
    matched = rule;
    ignored = !rule.negated;
  }
  return {
    path,
    relativePath,
    ignored,
    decision: matched == null ? 'unmatched' : ignored ? 'ignored' : 'included',
    source: policy.source,
    rule: matched,
    pathAccess,
  };
}

/** Convert an ignored evaluation into a content-free discovery diagnostic. */
export function aiIgnoreSuppressionDiagnostic(
  evaluation: AiIgnoreEvaluation,
  surface: AiIgnoreDiscoverySurface,
  kind: 'file' | 'directory',
): AiIgnoreSuppressionDiagnostic {
  return {
    code: 'aiignore-suppressed',
    severity: 'warning',
    path: evaluation.path,
    kind,
    surface,
    policyPath: evaluation.source.path,
    policyLine: evaluation.rule?.line,
    message: `Discovery suppressed ${kind} ${evaluation.path} by policy ${evaluation.source.path}`,
  };
}

function policySource(options: CompileAiIgnorePolicyOptions): AiIgnorePolicySource {
  const workspace = resolve(options.workspace);
  let realWorkspace: string;
  try {
    realWorkspace = fs.realpathSync(workspace);
  } catch (error) {
    throw new AiIgnorePolicyError(
      'WORKSPACE_UNAVAILABLE',
      `Workspace does not exist or cannot be resolved: ${workspace}`,
      { cause: error },
    );
  }
  return {
    path: resolve(options.sourcePath ?? resolve(workspace, AIIGNORE_FILE)),
    realPath: options.sourceRealPath == null ? undefined : resolve(options.sourceRealPath),
    workspace,
    realWorkspace,
    exists: options.sourceExists ?? options.sourceRealPath != null,
  };
}

function ruleMatches(
  rule: AiIgnoreRule,
  relativePath: string,
  kind: 'file' | 'directory' | undefined,
): boolean {
  const matcher = matchers.get(rule);
  const match = matcher?.exec(relativePath);
  if (match == null) return false;
  if (!rule.directoryOnly) return true;
  const exact = match.index + match[0].replace(/\/$/, '').length === relativePath.length;
  return !exact || kind === 'directory';
}

function compileMatcher(pattern: string, rootRelative: boolean): RegExp {
  const body = globToRegex(pattern);
  const prefix = rootRelative || pattern.includes('/') ? '^' : '(?:^|/)';
  return new RegExp(`${prefix}${body}(?:$|/)`);
}

function globToRegex(pattern: string): string {
  let output = '';
  for (let index = 0; index < pattern.length; index++) {
    const character = pattern[index] ?? '';
    if (character === '\\') {
      const next = pattern[index + 1];
      output += escapeRegex(next ?? '\\');
      if (next != null) index++;
      continue;
    }
    if (character === '*') {
      if (pattern[index + 1] === '*') {
        while (pattern[index + 1] === '*') index++;
        if (pattern[index + 1] === '/') {
          output += '(?:.*/)?';
          index++;
        } else {
          output += '.*';
        }
      } else {
        output += '[^/]*';
      }
      continue;
    }
    if (character === '?') {
      output += '[^/]';
      continue;
    }
    if (character === '[') {
      const end = findClassEnd(pattern, index + 1);
      if (end != null) {
        const value = pattern.slice(index + 1, end);
        const characterClass = compileCharacterClass(value);
        if (characterClass != null) {
          output += characterClass;
          index = end;
          continue;
        }
      }
    }
    output += escapeRegex(character);
  }
  return output;
}

function compileCharacterClass(value: string): string | undefined {
  const negated = value.startsWith('!') || value.startsWith('^');
  const body = negated ? value.slice(1) : value;
  if (body === '' || body.includes('/')) return undefined;
  const characterClass = `[${negated ? '^' : ''}${body.replace(/\\/g, '\\\\')}]`;
  try {
    new RegExp(characterClass);
    return characterClass;
  } catch {
    return undefined;
  }
}

function findClassEnd(pattern: string, start: number): number | undefined {
  for (let index = start; index < pattern.length; index++) {
    if (pattern[index] === ']' && index > start) return index;
  }
  return undefined;
}

function trimUnescapedTrailingSpaces(input: string): string {
  let end = input.length;
  while (end > 0 && input[end - 1] === ' ') {
    let slashes = 0;
    for (let index = end - 2; index >= 0 && input[index] === '\\'; index--) slashes++;
    if (slashes % 2 === 1) break;
    end--;
  }
  return input.slice(0, end);
}

function isUnescapedPrefix(input: string, prefix: string): boolean {
  return input.startsWith(prefix);
}

function endsWithUnescapedSlash(pattern: string): boolean {
  if (!pattern.endsWith('/')) return false;
  let slashes = 0;
  for (let index = pattern.length - 2; index >= 0 && pattern[index] === '\\'; index--) slashes++;
  return slashes % 2 === 0;
}

function normalizePattern(pattern: string): string {
  return pattern.replace(/^\.\//, '');
}

function candidatePath(workspace: string, input: string): string {
  if (isAbsolute(input)) return input;
  return `${workspace}${sep}${input}`;
}

function resolveCandidate(workspace: string, input: string): string {
  return isAbsolute(input) ? resolve(input) : resolve(workspace, input);
}

function normalizeRelativePath(path: string): string {
  return path.split(sep).join('/').replace(/^\.\//, '');
}

function existingKind(path: string): 'file' | 'directory' | undefined {
  try {
    const info = fs.statSync(path);
    return info.isDirectory() ? 'directory' : info.isFile() ? 'file' : undefined;
  } catch {
    return undefined;
  }
}

function existingRealPath(path: string): string | undefined {
  try {
    return fs.realpathSync(path);
  } catch {
    return undefined;
  }
}

function escapeRegex(character: string): string {
  return /[\\^$.*+?()[\]{}|/]/.test(character) ? `\\${character}` : character;
}

function isErrno(error: unknown, code: string): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === code;
}
