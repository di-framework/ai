import { isAbsolute, relative, resolve, sep } from 'node:path';
import { type AiIgnorePolicy, evaluateAiIgnorePath } from '../policy/aiignore.ts';

/**
 * Selects which direct file surfaces enforce an `.aiignore` policy.
 *
 * - `discovery`: directory listings hide ignored entries and reject ignored roots.
 * - `read`: discovery behavior plus Read and Edit rejection.
 * - `read-write`: read behavior plus Write rejection.
 */
export type AiIgnoreEnforcement = 'discovery' | 'read' | 'read-write';

/** A compiled policy and the enforcement level applied by a direct file tool. */
export interface AiIgnoreToolPolicy {
  readonly policy: AiIgnorePolicy;
  readonly enforcement: AiIgnoreEnforcement;
}

type AiIgnoreToolOperation = 'discovery' | 'read' | 'write';

/**
 * Return a content-free rejection for an ignored path, or `undefined` when the
 * operation is allowed. Callers must apply their filesystem sandbox first.
 */
export function aiIgnoreRejection(
  options: AiIgnoreToolPolicy | undefined,
  path: string,
  operation: AiIgnoreToolOperation,
  kind?: 'file' | 'directory',
): string | undefined {
  if (options == null || !enforces(options.enforcement, operation)) return undefined;
  if (!isWithinWorkspace(path, options.policy.source.workspace)) return undefined;

  const evaluation = evaluateAiIgnorePath(options.policy, path, { kind });
  if (!evaluation.pathAccess.ok) return evaluation.pathAccess.error;
  if (!evaluation.ignored) return undefined;

  const line = evaluation.rule == null ? '' : ` (rule line ${evaluation.rule.line})`;
  return `Error: Access denied for ${evaluation.path} by .aiignore policy ${evaluation.source.path}${line}`;
}

function enforces(enforcement: AiIgnoreEnforcement, operation: AiIgnoreToolOperation): boolean {
  if (operation === 'discovery') return true;
  if (operation === 'read') return enforcement === 'read' || enforcement === 'read-write';
  return enforcement === 'read-write';
}

function isWithinWorkspace(path: string, workspace: string): boolean {
  const resolvedPath = resolve(path);
  const resolvedWorkspace = resolve(workspace);
  if (resolvedPath === resolvedWorkspace) return true;
  const relationship = relative(resolvedWorkspace, resolvedPath);
  return (
    relationship !== '' &&
    relationship !== '..' &&
    !relationship.startsWith(`..${sep}`) &&
    !isAbsolute(relationship)
  );
}
