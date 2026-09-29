import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

/**
 * TypeScript session stays on 0.1.0 (`latest`).
 * Prebuilt CLIs publish as `@di-framework/ml@<workspace>-<os>-<arch>` from `publish-native-cli.ts`.
 */
const SKIP_PACKAGE = '@di-framework/ml';

type PackageManifest = {
  name?: string;
  version?: string;
};

/**
 * Set every workspace package to `version`, except {@link SKIP_PACKAGE}.
 * Returns how many manifests were rewritten.
 */
export function syncWorkspaceVersions(rootDir: string, version: string): number {
  const root = JSON.parse(readFileSync(join(rootDir, 'package.json'), 'utf8')) as {
    workspaces?: unknown;
  };
  const patterns = Array.isArray(root.workspaces)
    ? root.workspaces.filter((pattern): pattern is string => typeof pattern === 'string')
    : [];
  let updated = 0;

  for (const pattern of patterns) {
    const directories = pattern.endsWith('/*')
      ? readdirSync(join(rootDir, pattern.slice(0, -2)), { withFileTypes: true })
          .filter((entry) => entry.isDirectory())
          .map((entry) => join(pattern.slice(0, -2), entry.name))
      : [pattern];

    for (const directory of directories) {
      const pkgPath = join(rootDir, directory, 'package.json');
      if (!existsSync(pkgPath)) continue;
      const pkg = JSON.parse(readFileSync(pkgPath, 'utf8')) as PackageManifest;
      if (pkg.name === SKIP_PACKAGE) continue;
      if (pkg.version === version) continue;
      writeFileSync(pkgPath, `${JSON.stringify({ ...pkg, version }, null, 2)}\n`);
      updated += 1;
    }
  }

  return updated;
}

if (import.meta.main) {
  const version = process.env.NEXT;
  if (!version) {
    console.error('NEXT is not set');
    process.exit(1);
  }
  const updated = syncWorkspaceVersions(process.cwd(), version);
  console.log(`Set ${updated} workspace package version(s) to ${version}`);
}
