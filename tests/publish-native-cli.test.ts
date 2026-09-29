import { describe, expect, test } from 'bun:test';
import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  assetFileName,
  nativeVersion,
  parseRustcHost,
  platformForTarget,
  stageNativePackage,
} from '../scripts/publish-native-cli.ts';
import { missingPackedEntries, packPaths } from '../scripts/publish-workspace.ts';

describe('native cli publish versions', () => {
  test('maps rustc hosts to os-arch prereleases', () => {
    expect(platformForTarget('aarch64-apple-darwin').suffix).toBe('darwin-aarch64');
    expect(platformForTarget('x86_64-unknown-linux-gnu').suffix).toBe('linux-x64');
    expect(platformForTarget('aarch64-unknown-linux-gnu').suffix).toBe('linux-aarch64');
    expect(platformForTarget('x86_64-pc-windows-msvc').suffix).toBe('windows-x64');
    expect(nativeVersion('6.0.1', 'darwin-aarch64')).toBe('6.0.1-darwin-aarch64');
    expect(assetFileName(platformForTarget('x86_64-pc-windows-msvc'))).toBe(
      'di-ml-windows-x64.exe',
    );
    expect(parseRustcHost('release: 1.97.1\nhost: aarch64-apple-darwin\n')).toBe(
      'aarch64-apple-darwin',
    );
  });

  test('rejects versions semver cannot publish', () => {
    expect(() => nativeVersion('6.0.1-darwin-aarch64', 'linux-x64')).toThrow('MAJOR.MINOR.PATCH');
    expect(() => nativeVersion('6.0.1', 'linux-x86_64')).toThrow('prerelease');
    expect(() => platformForTarget('wasm32-unknown-unknown')).toThrow('unsupported rustc host');
  });

  test('stages a pack whose version is the workspace version plus os and arch', () => {
    const root = mkdtempSync(join(tmpdir(), 'native-cli-root-'));
    const stage = mkdtempSync(join(tmpdir(), 'native-cli-stage-'));
    const binary = join(root, 'di-ml');
    writeFileSync(join(root, 'package.json'), JSON.stringify({ version: '6.0.1' }));
    writeFileSync(join(root, 'LICENSE'), 'license\n');
    writeFileSync(binary, '#!/bin/sh\necho di-ml\n');
    chmodSync(binary, 0o755);

    const platform = platformForTarget('aarch64-apple-darwin');
    const manifest = stageNativePackage({
      stageDir: stage,
      rootDir: root,
      baseVersion: '6.0.1',
      platform,
      binaryPath: binary,
    });

    expect(manifest.name).toBe('@di-framework/ml');
    expect(manifest.version).toBe('6.0.1-darwin-aarch64');
    expect(manifest.os).toEqual(['darwin']);
    expect(manifest.cpu).toEqual(['arm64']);
    expect(manifest.bin).toEqual({ 'di-ml': 'bin/di-ml' });
    expect(manifest.files).toContain('bin');
    expect(manifest.files).toContain('LICENSE');
    const written = JSON.parse(readFileSync(join(stage, 'package.json'), 'utf8')) as {
      version: string;
    };
    expect(written.version).toBe('6.0.1-darwin-aarch64');
    expect(missingPackedEntries(manifest.files, packPaths(stage))).toEqual([]);
  });
});
