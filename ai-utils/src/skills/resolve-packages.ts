import { readFileSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { expandUserPath } from '../sandbox/paths.ts';

/**
 * Resolve npm package names or paths to skill root directories.
 *
 * Looks at {@code package.json} {@code skills} (string or string[]), then
 * {@code .agents/skills}, then {@code skills} under the package root.
 */
export function resolveSkillPackageDirectories(
  packages: readonly string[],
  fromDirectory: string = process.cwd(),
): string[] {
  const out: string[] = [];
  for (const spec of packages) {
    out.push(...skillDirectoriesForPackage(spec, fromDirectory));
  }
  return out;
}

function skillDirectoriesForPackage(spec: string, fromDirectory: string): string[] {
  const root = resolvePackageRoot(spec, fromDirectory);
  const declared = readPackageSkillsField(root);
  const declaredExisting = declared.filter((dir) => isExistingDirectory(dir));
  if (declaredExisting.length > 0) return declaredExisting;
  const neutral = join(root, '.agents', 'skills');
  if (isExistingDirectory(neutral)) return [neutral];
  const conventional = join(root, 'skills');
  return isExistingDirectory(conventional) ? [conventional] : [];
}

function resolvePackageRoot(spec: string, fromDirectory: string): string {
  const expanded = expandUserPath(spec);
  if (expanded.startsWith('.') || isAbsolute(expanded)) {
    return resolve(fromDirectory, expanded);
  }
  const require = createRequire(join(fromDirectory, 'package.json'));
  try {
    return dirname(require.resolve(`${expanded}/package.json`));
  } catch {
    throw new Error(`Cannot resolve skill package "${spec}" from ${fromDirectory}`);
  }
}

function isExistingDirectory(dir: string): boolean {
  try {
    return statSync(dir).isDirectory();
  } catch {
    return false;
  }
}

function readPackageSkillsField(packageRoot: string): string[] {
  const pkgPath = join(packageRoot, 'package.json');
  let parsed: { skills?: unknown };
  try {
    parsed = JSON.parse(readFileSync(pkgPath, 'utf8')) as { skills?: unknown };
  } catch {
    return [];
  }
  const field = parsed.skills;
  const entries = Array.isArray(field) ? field : typeof field === 'string' ? [field] : [];
  return entries
    .filter((entry): entry is string => typeof entry === 'string' && entry.trim().length > 0)
    .map((entry) => resolve(packageRoot, entry));
}
