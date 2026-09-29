import { describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { assertPathAllowed, expandUserPath, uniqueResolvedRoots } from '../src/index.ts';
import { errorMessage, nodeErrnoCode } from '../src/sandbox/fs-error.ts';

describe('assertPathAllowed', () => {
  test('allows a file inside an allowed root', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-sand-'));
    const file = join(root, 'ok.txt');
    writeFileSync(file, 'hi');
    const result = assertPathAllowed(file, [root]);
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.path).toBe(file);
  });

  test('rejects a path outside the root', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-sand-'));
    const result = assertPathAllowed('/etc/passwd', [root]);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toContain('outside the allowed directories');
  });

  test('rejects raw .. components even when they would normalize inside', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-sand-'));
    const result = assertPathAllowed(`${root}/sub/../ok.txt`, [root]);
    expect(result.ok).toBe(false);
  });

  test('rejects a sibling prefix that is not the root', () => {
    const parent = mkdtempSync(join(tmpdir(), 'ai-utils-sand-'));
    const allowed = join(parent, 'allowed');
    const evil = join(parent, 'allowed-evil', 'secret');
    mkdirSync(allowed);
    mkdirSync(join(parent, 'allowed-evil'));
    writeFileSync(evil, 'noop');
    const result = assertPathAllowed(evil, [allowed]);
    expect(result.ok).toBe(false);
  });

  test('rejects a symlink that points outside the root', () => {
    const parent = mkdtempSync(join(tmpdir(), 'ai-utils-sand-'));
    const allowed = join(parent, 'allowed');
    const outside = join(parent, 'outside.txt');
    mkdirSync(allowed);
    writeFileSync(outside, 'secret');
    const link = join(allowed, 'escape');
    symlinkSync(outside, link);
    const result = assertPathAllowed(link, [allowed]);
    expect(result.ok).toBe(false);
  });

  test('rejects an empty path and empty allow-list', () => {
    expect(assertPathAllowed('', ['/tmp']).ok).toBe(false);
    expect(assertPathAllowed('/tmp/x', []).ok).toBe(false);
  });

  test('expandUserPath and uniqueResolvedRoots', () => {
    expect(expandUserPath('~/skills').startsWith(expandUserPath('~'))).toBe(true);
    const roots = uniqueResolvedRoots(['/tmp/a', '/tmp/a/']);
    expect(roots).toHaveLength(1);
  });
});

describe('nodeErrnoCode', () => {
  test('reads string codes and ignores everything else', () => {
    expect(nodeErrnoCode(null)).toBeUndefined();
    expect(nodeErrnoCode('enoent')).toBeUndefined();
    expect(nodeErrnoCode({})).toBeUndefined();
    expect(nodeErrnoCode({ code: 2 })).toBeUndefined();
    expect(nodeErrnoCode({ code: 'ENOENT' })).toBe('ENOENT');
    expect(errorMessage(new Error('boom'))).toBe('boom');
    expect(errorMessage('plain')).toBe('plain');
  });
});
