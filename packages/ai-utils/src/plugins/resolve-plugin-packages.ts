import { readFileSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { expandUserPath } from '../sandbox/paths.ts';

/**
 * Resolve npm package names or paths to plugin root directories.
 *
 * Looks at {@code package.json} {@code plugins} (string or string[]), then
 * {@code .agents/plugins}, then {@code plugins} under the package root, then
 * a package-root {@code plugin.json} for single-plugin packages.
 */
export function resolvePluginPackageDirectories(
  packages: readonly string[],
  fromDirectory: string = process.cwd(),
): string[] {
  const out: string[] = [];
  for (const spec of packages) {
    out.push(...pluginDirectoriesForPackage(spec, fromDirectory));
  }
  return out;
}

function pluginDirectoriesForPackage(spec: string, fromDirectory: string): string[] {
  const root = resolvePackageRoot(spec, fromDirectory);
  const declared = readPackagePluginsField(root);
  const declaredExisting = declared.filter((dir) => isExistingDirectory(dir));
  if (declaredExisting.length > 0) return declaredExisting;
  const neutral = join(root, '.agents', 'plugins');
  if (isExistingDirectory(neutral)) return [neutral];
  const conventional = join(root, 'plugins');
  if (isExistingDirectory(conventional)) return [conventional];
  // Single-plugin packages publish {@code plugin.json} at the package root
  // (for example {@code @di-framework/plugin}).
  return isExistingFile(join(root, 'plugin.json')) ? [root] : [];
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
    throw new Error(`Cannot resolve plugin package "${spec}" from ${fromDirectory}`);
  }
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

function readPackagePluginsField(packageRoot: string): string[] {
  const pkgPath = join(packageRoot, 'package.json');
  let parsed: { plugins?: unknown };
  try {
    parsed = JSON.parse(readFileSync(pkgPath, 'utf8')) as { plugins?: unknown };
  } catch {
    return [];
  }
  const field = parsed.plugins;
  const entries = Array.isArray(field) ? field : typeof field === 'string' ? [field] : [];
  return entries
    .filter((entry): entry is string => typeof entry === 'string' && entry.trim().length > 0)
    .map((entry) => resolve(packageRoot, entry));
}
