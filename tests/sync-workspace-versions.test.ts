import { describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { syncWorkspaceVersions } from '../scripts/sync-workspace-versions.ts';

describe('sync workspace versions', () => {
  test('bumps workspace packages and leaves @di-framework/ml alone', () => {
    const root = mkdtempSync(join(tmpdir(), 'sync-workspace-'));
    mkdirSync(join(root, 'packages', 'ai'), { recursive: true });
    mkdirSync(join(root, 'packages', 'ml', 'infer'), { recursive: true });
    writeFileSync(
      join(root, 'package.json'),
      JSON.stringify({ private: true, workspaces: ['packages/*', 'packages/ml/infer'] }),
    );
    writeFileSync(
      join(root, 'packages', 'ai', 'package.json'),
      JSON.stringify({ name: '@di-framework/ai', version: '6.0.1' }),
    );
    writeFileSync(
      join(root, 'packages', 'ml', 'infer', 'package.json'),
      JSON.stringify({ name: '@di-framework/ml', version: '0.1.0' }),
    );

    expect(syncWorkspaceVersions(root, '6.0.2')).toBe(1);

    const ai = JSON.parse(readFileSync(join(root, 'packages', 'ai', 'package.json'), 'utf8')) as {
      version: string;
    };
    const ml = JSON.parse(
      readFileSync(join(root, 'packages', 'ml', 'infer', 'package.json'), 'utf8'),
    ) as { version: string };
    expect(ai.version).toBe('6.0.2');
    expect(ml.version).toBe('0.1.0');
  });
});
