import { describe, expect, spyOn, test } from 'bun:test';
import * as fs from 'node:fs';
import { mkdirSync, mkdtempSync, realpathSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  AiIgnorePolicyError,
  compileAiIgnorePolicy,
  evaluateAiIgnorePath,
  loadAiIgnorePolicy,
} from '../src/policy/index.ts';

function fixture(): { root: string; workspace: string } {
  const root = mkdtempSync(join(tmpdir(), 'ai-utils-aiignore-'));
  const workspace = join(root, 'workspace');
  mkdirSync(workspace);
  return { root, workspace };
}

describe('.aiignore policy', () => {
  test('compiles comments, escapes, root and directory markers with rule provenance', () => {
    const { workspace } = fixture();
    const sourcePath = join(workspace, 'supplied.aiignore');
    const policy = compileAiIgnorePolicy(
      [
        '# comment',
        '',
        String.raw`\#literal`,
        String.raw`\!literal`,
        '/root.txt',
        'cache/',
        '!keep.log',
      ].join('\r\n'),
      { workspace, sourcePath },
    );

    expect(policy.source).toEqual({
      path: sourcePath,
      realPath: undefined,
      workspace,
      realWorkspace: realpathSync(workspace),
      exists: false,
    });
    expect(
      policy.rules.map(
        ({ line, original, pattern, negated, directoryOnly, rootRelative, source }) => ({
          line,
          original,
          pattern,
          negated,
          directoryOnly,
          rootRelative,
          sourcePath: source.path,
        }),
      ),
    ).toEqual([
      {
        line: 3,
        original: String.raw`\#literal`,
        pattern: String.raw`\#literal`,
        negated: false,
        directoryOnly: false,
        rootRelative: false,
        sourcePath,
      },
      {
        line: 4,
        original: String.raw`\!literal`,
        pattern: String.raw`\!literal`,
        negated: false,
        directoryOnly: false,
        rootRelative: false,
        sourcePath,
      },
      {
        line: 5,
        original: '/root.txt',
        pattern: 'root.txt',
        negated: false,
        directoryOnly: false,
        rootRelative: true,
        sourcePath,
      },
      {
        line: 6,
        original: 'cache/',
        pattern: 'cache',
        negated: false,
        directoryOnly: true,
        rootRelative: false,
        sourcePath,
      },
      {
        line: 7,
        original: '!keep.log',
        pattern: 'keep.log',
        negated: true,
        directoryOnly: false,
        rootRelative: false,
        sourcePath,
      },
    ]);
  });

  test('supports wildcards, root-relative rules, directories, and normalized missing paths', () => {
    const { workspace } = fixture();
    const policy = compileAiIgnorePolicy(
      ['*.log', '/root.txt', 'docs/*.md', 'src/**/generated?.[jt]s', 'cache/'].join('\n'),
      { workspace },
    );

    expect(evaluateAiIgnorePath(policy, 'nested/output.log').decision).toBe('ignored');
    expect(evaluateAiIgnorePath(policy, './root.txt').decision).toBe('ignored');
    expect(evaluateAiIgnorePath(policy, 'nested/root.txt').decision).toBe('unmatched');
    expect(evaluateAiIgnorePath(policy, 'docs/guide.md').decision).toBe('ignored');
    expect(evaluateAiIgnorePath(policy, 'nested/docs/guide.md').decision).toBe('unmatched');
    expect(evaluateAiIgnorePath(policy, 'src/generated1.js').decision).toBe('ignored');
    expect(evaluateAiIgnorePath(policy, 'src/deep/generatedx.ts').decision).toBe('ignored');
    expect(evaluateAiIgnorePath(policy, 'src/deep/generatedxx.ts').decision).toBe('unmatched');
    expect(evaluateAiIgnorePath(policy, 'cache', { kind: 'file' }).decision).toBe('unmatched');
    expect(evaluateAiIgnorePath(policy, 'cache', { kind: 'directory' }).decision).toBe('ignored');
    expect(evaluateAiIgnorePath(policy, 'cache/entry.txt').decision).toBe('ignored');
  });

  test('uses the last matching rule and reports the effective rule and policy source', () => {
    const { workspace } = fixture();
    const policy = compileAiIgnorePolicy(
      ['*.log', '!keep.log', 'keep.log', '!nested/keep.log'].join('\n'),
      { workspace },
    );

    const included = evaluateAiIgnorePath(policy, 'nested/keep.log');
    expect(included).toMatchObject({
      relativePath: 'nested/keep.log',
      ignored: false,
      decision: 'included',
      source: policy.source,
      rule: {
        line: 4,
        original: '!nested/keep.log',
        pattern: 'nested/keep.log',
        negated: true,
      },
      pathAccess: { ok: true },
    });

    const ignored = evaluateAiIgnorePath(policy, 'keep.log');
    expect(ignored.decision).toBe('ignored');
    expect(ignored.rule?.line).toBe(3);
    expect(evaluateAiIgnorePath(policy, 'readme.md')).toMatchObject({
      ignored: false,
      decision: 'unmatched',
      rule: undefined,
    });
  });

  test('preserves escaped leading markers and escaped trailing spaces', () => {
    const { workspace } = fixture();
    const policy = compileAiIgnorePolicy(
      [
        'plain.txt   ',
        String.raw`space\ `,
        String.raw`\#literal`,
        String.raw`\!literal`,
        '!',
        '/',
      ].join('\n'),
      { workspace },
    );

    expect(policy.rules).toHaveLength(4);
    expect(evaluateAiIgnorePath(policy, 'plain.txt').decision).toBe('ignored');
    expect(evaluateAiIgnorePath(policy, 'space ').decision).toBe('ignored');
    expect(evaluateAiIgnorePath(policy, '#literal').decision).toBe('ignored');
    expect(evaluateAiIgnorePath(policy, '!literal').decision).toBe('ignored');
  });

  test('loads only the root policy and keeps the policy file itself available', () => {
    const { workspace } = fixture();
    const sourcePath = join(workspace, '.aiignore');
    writeFileSync(sourcePath, '*\n');
    mkdirSync(join(workspace, 'nested'));
    writeFileSync(join(workspace, 'nested', '.aiignore'), '!visible.txt\n');
    symlinkSync(sourcePath, join(workspace, 'policy-alias'));

    const policy = loadAiIgnorePolicy({ workspace });
    expect(policy.source).toEqual({
      path: sourcePath,
      realPath: realpathSync(sourcePath),
      workspace,
      realWorkspace: realpathSync(workspace),
      exists: true,
    });
    expect(policy.rules).toHaveLength(1);
    expect(evaluateAiIgnorePath(policy, sourcePath)).toMatchObject({
      relativePath: '.aiignore',
      ignored: false,
      decision: 'policy-file',
      source: policy.source,
    });
    expect(evaluateAiIgnorePath(policy, 'policy-alias').decision).toBe('policy-file');
    expect(evaluateAiIgnorePath(policy, 'nested/.aiignore').decision).toBe('ignored');
    expect(evaluateAiIgnorePath(policy, 'nested/visible.txt').decision).toBe('ignored');
  });

  test('returns an empty policy when the root policy is absent', () => {
    const { workspace } = fixture();
    const policy = loadAiIgnorePolicy({ workspace });

    expect(policy.source).toMatchObject({
      path: join(workspace, '.aiignore'),
      realPath: undefined,
      exists: false,
    });
    expect(policy.rules).toEqual([]);
    expect(evaluateAiIgnorePath(policy, 'anything.txt').decision).toBe('unmatched');
  });

  test('applies the filesystem sandbox before policy matching', () => {
    const { root, workspace } = fixture();
    const outside = join(root, 'outside');
    const outsideFile = join(outside, 'secret.txt');
    mkdirSync(outside);
    writeFileSync(outsideFile, 'secret');
    symlinkSync(outsideFile, join(workspace, 'secret-link'));
    const policy = compileAiIgnorePolicy('*\n', { workspace });

    for (const input of ['../outside/secret.txt', join(workspace, 'secret-link')]) {
      const evaluation = evaluateAiIgnorePath(policy, input);
      expect(evaluation).toMatchObject({
        ignored: false,
        decision: 'sandbox-denied',
        source: policy.source,
        pathAccess: { ok: false },
      });
      expect(evaluation.rule).toBeUndefined();
    }
  });

  test('supports a symlinked workspace without weakening its real boundary', () => {
    const { root, workspace } = fixture();
    const linkedWorkspace = join(root, 'workspace-link');
    writeFileSync(join(workspace, '.aiignore'), '*.tmp\n');
    symlinkSync(workspace, linkedWorkspace, 'dir');

    const policy = loadAiIgnorePolicy({ workspace: linkedWorkspace });
    expect(policy.source.realWorkspace).toBe(realpathSync(workspace));
    expect(evaluateAiIgnorePath(policy, 'nested/value.tmp')).toMatchObject({
      relativePath: 'nested/value.tmp',
      ignored: true,
      decision: 'ignored',
    });
    expect(evaluateAiIgnorePath(policy, join(linkedWorkspace, '.aiignore')).decision).toBe(
      'policy-file',
    );
  });

  test('loads an in-workspace policy symlink and preserves canonical provenance', () => {
    const { workspace } = fixture();
    const policyTarget = join(workspace, 'config', 'policy');
    mkdirSync(join(workspace, 'config'));
    writeFileSync(policyTarget, '*.generated\n');
    symlinkSync(policyTarget, join(workspace, '.aiignore'));

    const policy = loadAiIgnorePolicy({ workspace });
    expect(policy.source).toMatchObject({
      path: join(workspace, '.aiignore'),
      realPath: realpathSync(policyTarget),
      exists: true,
    });
    expect(evaluateAiIgnorePath(policy, '.aiignore').decision).toBe('policy-file');
    expect(evaluateAiIgnorePath(policy, 'value.generated').decision).toBe('ignored');
  });

  test('rejects a root policy symlink that escapes the workspace', () => {
    const { root, workspace } = fixture();
    const outsidePolicy = join(root, 'outside.aiignore');
    writeFileSync(outsidePolicy, '*\n');
    symlinkSync(outsidePolicy, join(workspace, '.aiignore'));

    expect(() => loadAiIgnorePolicy({ workspace })).toThrow(AiIgnorePolicyError);
    try {
      loadAiIgnorePolicy({ workspace });
    } catch (error) {
      expect(error).toMatchObject({ code: 'POLICY_OUTSIDE_WORKSPACE' });
    }
  });

  test('reports unavailable workspaces and unreadable policies with typed errors', () => {
    const { root, workspace } = fixture();
    const absent = join(root, 'absent');
    expect(() => compileAiIgnorePolicy('*', { workspace: absent })).toThrow(
      expect.objectContaining({ code: 'WORKSPACE_UNAVAILABLE' }),
    );

    const sourcePath = join(workspace, '.aiignore');
    writeFileSync(sourcePath, '*\n');
    const originalRead = fs.readFileSync;
    const read = spyOn(fs, 'readFileSync').mockImplementation(((path: fs.PathOrFileDescriptor) => {
      if (path === realpathSync(sourcePath)) {
        throw Object.assign(new Error('denied'), { code: 'EACCES' });
      }
      return originalRead(path, 'utf8');
    }) as unknown as typeof fs.readFileSync);
    try {
      expect(() => loadAiIgnorePolicy({ workspace })).toThrow(
        expect.objectContaining({ code: 'POLICY_UNREADABLE' }),
      );
    } finally {
      read.mockRestore();
    }
  });

  test('treats malformed character classes and a trailing escape literally', () => {
    const { workspace } = fixture();
    const policy = compileAiIgnorePolicy('broken[\nrange[z-a]\nslash\\\\', { workspace });

    expect(evaluateAiIgnorePath(policy, 'broken[').decision).toBe('ignored');
    expect(evaluateAiIgnorePath(policy, 'range[z-a]').decision).toBe('ignored');
    expect(evaluateAiIgnorePath(policy, 'slash\\').decision).toBe('ignored');
  });
});
