import * as fs from 'node:fs';
import { homedir } from 'node:os';
import { isAbsolute, join, relative, resolve, sep } from 'node:path';
import { errorMessage, nodeErrnoCode } from './fs-error.ts';

export interface PathAccessOk {
  readonly ok: true;
  readonly path: string;
}

export interface PathAccessDenied {
  readonly ok: false;
  readonly error: string;
}

export type PathAccessResult = PathAccessOk | PathAccessDenied;

/**
 * Expand a leading {@code ~} to the user home directory.
 */
export function expandUserPath(input: string): string {
  const trimmed = input.trim();
  if (trimmed === '~') return homedir();
  if (trimmed.startsWith(`~${sep}`) || trimmed.startsWith('~/')) {
    return join(homedir(), trimmed.slice(2));
  }
  return trimmed;
}

export function uniqueResolvedRoots(roots: readonly string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const root of roots) {
    const resolved = resolve(expandUserPath(root));
    if (seen.has(resolved)) continue;
    seen.add(resolved);
    out.push(resolved);
  }
  return out;
}

/**
 * Spring FileSystemTools-style containment: reject raw {@code ..}, require
 * normalized containment, then realpath existing components (deny dangling
 * symlinks that cannot be verified).
 */
export function assertPathAllowed(
  filePath: string,
  allowedRoots: readonly string[],
): PathAccessResult {
  if (!filePath?.trim()) {
    return { ok: false, error: 'Error: Path is required' };
  }
  if (allowedRoots.length === 0) {
    return { ok: false, error: 'Error: No allowed directories configured' };
  }

  if (hasDotDotComponent(filePath)) {
    return {
      ok: false,
      error: `Error: Access denied. Path is outside the allowed directories: ${filePath}`,
    };
  }

  let target: string;
  try {
    target = resolve(expandUserPath(filePath));
  } catch (error) {
    return { ok: false, error: `Error: Invalid path: ${errorMessage(error)}` };
  }

  const roots = uniqueResolvedRoots(allowedRoots);

  for (const allowed of roots) {
    if (!isContained(target, allowed)) continue;

    try {
      const realAllowed = fs.realpathSync(allowed);
      const existing = nearestExisting(target);
      if (existing == null) {
        return { ok: true, path: target };
      }
      let realExisting: string;
      try {
        realExisting = fs.realpathSync(existing);
      } catch {
        return {
          ok: false,
          error: `Error: Access denied. Cannot resolve path (possible dangling symlink): ${filePath}`,
        };
      }
      if (isContained(realExisting, realAllowed)) {
        return { ok: true, path: target };
      }
    } catch (error) {
      if (nodeErrnoCode(error) === 'ENOENT') {
        return { ok: true, path: target };
      }
      return { ok: false, error: `Error validating path: ${errorMessage(error)}` };
    }
  }

  return {
    ok: false,
    error: `Error: Access denied. Path is outside the allowed directories: ${filePath}`,
  };
}

function hasDotDotComponent(filePath: string): boolean {
  return filePath.split(/[/\\]/).includes('..');
}

function isContained(target: string, root: string): boolean {
  if (target === root) return true;
  const rel = relative(root, target);
  return rel !== '' && !rel.startsWith(`..${sep}`) && rel !== '..' && !isAbsolute(rel);
}

function nearestExisting(target: string): string | undefined {
  let current = target;
  let parent = resolve(current, '..');
  while (current !== parent) {
    try {
      fs.lstatSync(current);
      return current;
    } catch {
      current = parent;
      parent = resolve(current, '..');
    }
  }
  try {
    fs.lstatSync(current);
    return current;
  } catch {
    return undefined;
  }
}
