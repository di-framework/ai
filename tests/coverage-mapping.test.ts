import { describe, expect, it } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import {
  calculatePackageMetrics,
  generateShieldBadgeJson,
  getPackageSlugFromPath,
  getWorkspacePackages,
  isSourceFile,
  parseLcov,
  writeShieldBadgeFiles,
} from '../scripts/coverage-mapping';

const repoRoot = resolve(import.meta.dir, '..');

function writePackage(root: string, relPath: string, name: string): void {
  const dir = join(root, relPath);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'package.json'), JSON.stringify({ name }));
}

describe('coverage badge mapping', () => {
  it('discovers every @di-framework package, including nested ones', () => {
    const packages = getWorkspacePackages(repoRoot);
    expect(packages.map((pkg) => pkg.name)).toEqual([
      '@di-framework/ai',
      '@di-framework/ai-utils',
      '@di-framework/ml',
    ]);
    expect(packages.map((pkg) => pkg.slug)).toEqual(['ai', 'ai-utils', 'ml']);
    expect(packages.find((pkg) => pkg.slug === 'ml')?.relPath).toBe('packages/ml/infer');
  });

  it('maps source files onto package slugs and ignores tests', () => {
    getWorkspacePackages(repoRoot);
    expect(getPackageSlugFromPath('packages/ai/src/index.ts')).toBe('ai');
    expect(getPackageSlugFromPath('packages/ai-utils/src/index.ts')).toBe('ai-utils');
    expect(getPackageSlugFromPath('packages/ml/infer/src/index.ts')).toBe('ml');
    expect(isSourceFile('packages/ai/src/index.ts')).toBe(true);
    expect(isSourceFile('packages/ai/tests/preload-wasm-mock.ts')).toBe(false);
    expect(isSourceFile('scripts/publish-workspace.ts')).toBe(false);
  });

  it('skips node_modules', () => {
    const root = mkdtempSync(join(tmpdir(), 'cov-discover-'));
    writePackage(root, 'packages/ai', '@di-framework/ai');
    writePackage(root, 'packages/ml/infer', '@di-framework/ml');
    writePackage(root, 'node_modules/@di-framework/core', '@di-framework/core');

    const packages = getWorkspacePackages(root);
    expect(packages.map((pkg) => pkg.slug)).toEqual(['ai', 'ml']);
    expect(getPackageSlugFromPath(join(root, 'packages/ml/infer/src/index.ts'))).toBe('ml');
  });

  it('writes Shields endpoint JSON for each discovered package', () => {
    const root = mkdtempSync(join(tmpdir(), 'cov-pkgs-'));
    writePackage(root, 'packages/ai', '@di-framework/ai');
    writePackage(root, 'packages/ai-utils', '@di-framework/ai-utils');
    const packages = getWorkspacePackages(root);
    const lcov = `
SF:${join(root, 'packages/ai/src/index.ts')}
DA:1,1
DA:2,1
end_of_record
SF:${join(root, 'packages/ai-utils/src/index.ts')}
DA:1,1
end_of_record
`;
    const metrics = calculatePackageMetrics(packages, parseLcov(lcov));
    const ai = metrics.find((metric) => metric.slug === 'ai');
    if (!ai) throw new Error('missing package metric');
    expect(ai.badgeMessage).toBe('100%');
    expect(generateShieldBadgeJson(ai)).toEqual({
      schemaVersion: 1,
      label: 'line coverage',
      message: '100%',
      color: 'brightgreen',
    });

    const outDir = join(root, 'coverage', 'badges');
    const written = writeShieldBadgeFiles(metrics, outDir);
    expect(written.map((file) => file.slice(outDir.length + 1)).sort()).toEqual([
      'ai-utils.json',
      'ai.json',
    ]);
  });
});
