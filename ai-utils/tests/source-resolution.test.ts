import { describe, expect, spyOn, test } from 'bun:test';
import * as fs from 'node:fs';
import { mkdirSync, mkdtempSync, realpathSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { resolveAgentSources } from '../src/index.ts';

function fixture(): { workspace: string; userDirectory: string; allowed: string } {
  const root = mkdtempSync(join(tmpdir(), 'ai-utils-sources-'));
  const workspace = join(root, 'workspace');
  const userDirectory = join(root, 'home');
  const allowed = join(root, 'allowed');
  mkdirSync(workspace);
  mkdirSync(userDirectory);
  mkdirSync(allowed);
  return { workspace, userDirectory, allowed };
}

describe('resolveAgentSources', () => {
  test('resolves workspace-relative and user candidates with provenance and precedence', () => {
    const { workspace, userDirectory } = fixture();
    const workspaceSource = join(workspace, 'AGENTS.md');
    const userSource = join(userDirectory, '.agents', 'skills');
    writeFileSync(workspaceSource, '# workspace');
    mkdirSync(userSource, { recursive: true });

    const result = resolveAgentSources(
      [
        { path: 'AGENTS.md', origin: 'workspace', kind: 'file' },
        { path: '~/.agents/skills', origin: 'user', kind: 'directory' },
      ],
      { workspace, userDirectory },
    );

    expect(result.diagnostics).toEqual([]);
    expect(result.sources).toEqual([
      {
        path: workspaceSource,
        realPath: realpathSync(workspaceSource),
        origin: 'workspace',
        precedence: 0,
        kind: 'file',
      },
      {
        path: userSource,
        realPath: realpathSync(userSource),
        origin: 'user',
        precedence: 1,
        kind: 'directory',
      },
    ]);
  });

  test('keeps the first real path and diagnoses direct and symlink duplicates', () => {
    const { workspace } = fixture();
    const source = join(workspace, 'skills');
    const alias = join(workspace, 'skills-alias');
    mkdirSync(source);
    symlinkSync(source, alias, 'dir');

    const result = resolveAgentSources(
      [
        { path: source, origin: 'explicit', kind: 'directory' },
        { path: './skills', origin: 'workspace', kind: 'directory' },
        { path: alias, origin: 'explicit', kind: 'directory' },
      ],
      { workspace },
    );

    expect(result.sources).toHaveLength(1);
    expect(result.sources[0]?.precedence).toBe(0);
    expect(result.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
      'source-duplicate',
      'source-duplicate',
    ]);
    expect(result.diagnostics[1]?.duplicateOf).toBe(source);
  });

  test('rejects lexical and symlink escapes from workspace and allowed roots', () => {
    const { workspace, allowed } = fixture();
    const allowedFile = join(allowed, 'shared.md');
    const outside = mkdtempSync(join(tmpdir(), 'ai-utils-outside-'));
    const outsideFile = join(outside, 'secret.md');
    writeFileSync(outsideFile, 'secret');
    writeFileSync(allowedFile, 'shared');
    const workspaceAlias = join(workspace, 'escaped.md');
    const allowedAlias = join(allowed, 'escaped.md');
    symlinkSync(outsideFile, workspaceAlias);
    symlinkSync(outsideFile, allowedAlias);

    const result = resolveAgentSources(
      [
        { path: allowedFile, origin: 'explicit', kind: 'file' },
        { path: outsideFile, origin: 'workspace', kind: 'file' },
        { path: workspaceAlias, origin: 'workspace', kind: 'file' },
        { path: allowedAlias, origin: 'explicit', kind: 'file' },
      ],
      { workspace, allowedDirectories: [allowed] },
    );

    expect(result.sources).toEqual([
      {
        path: allowedFile,
        realPath: realpathSync(allowedFile),
        origin: 'explicit',
        precedence: 0,
        kind: 'file',
      },
    ]);
    expect(result.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
      'source-outside-boundary',
      'source-outside-boundary',
      'source-outside-boundary',
    ]);
    expect(result.diagnostics.map((diagnostic) => diagnostic.precedence)).toEqual([1, 2, 3]);
  });

  test('diagnoses missing, broken, wrong-kind, and unreadable candidates deterministically', () => {
    const { workspace } = fixture();
    const broken = join(workspace, 'broken');
    const directory = join(workspace, 'directory');
    const unreadable = join(workspace, 'unreadable.md');
    symlinkSync(join(workspace, 'absent-target'), broken);
    mkdirSync(directory);
    writeFileSync(unreadable, 'private');
    const unreadableRealPath = realpathSync(unreadable);
    const originalAccess = fs.accessSync;
    const access = spyOn(fs, 'accessSync').mockImplementation((path, mode) => {
      if (path === unreadableRealPath) {
        throw Object.assign(new Error('denied'), { code: 'EACCES' });
      }
      return originalAccess(path, mode);
    });

    try {
      const result = resolveAgentSources(
        [
          { path: 'missing', origin: 'workspace' },
          { path: broken, origin: 'workspace' },
          { path: directory, origin: 'workspace', kind: 'file' },
          { path: unreadable, origin: 'workspace', kind: 'file' },
        ],
        { workspace },
      );

      expect(result.sources).toEqual([]);
      expect(result.diagnostics.map(({ code, precedence }) => ({ code, precedence }))).toEqual([
        { code: 'source-missing', precedence: 0 },
        { code: 'source-broken-symlink', precedence: 1 },
        { code: 'source-kind-mismatch', precedence: 2 },
        { code: 'source-unreadable', precedence: 3 },
      ]);
      expect(result.diagnostics.map((diagnostic) => diagnostic.message)).toEqual([
        `Source does not exist: ${join(workspace, 'missing')}`,
        `Source is a broken symlink: ${broken}`,
        `Source must be a file: ${directory}`,
        `Source is unreadable: ${unreadable}`,
      ]);
    } finally {
      access.mockRestore();
    }
  });

  test('does not let a symlinked workspace root weaken its real boundary', () => {
    const { workspace } = fixture();
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-workspace-link-'));
    const linkedWorkspace = join(root, 'workspace-link');
    const source = join(workspace, 'AGENTS.md');
    writeFileSync(source, '# instructions');
    symlinkSync(workspace, linkedWorkspace, 'dir');

    const result = resolveAgentSources(
      [{ path: join(linkedWorkspace, 'AGENTS.md'), origin: 'workspace', kind: 'file' }],
      { workspace: linkedWorkspace },
    );

    expect(result.diagnostics).toEqual([]);
    expect(result.sources[0]).toMatchObject({
      path: join(linkedWorkspace, 'AGENTS.md'),
      realPath: realpathSync(source),
    });
  });

  test('diagnoses a source that becomes unresolvable after stat', () => {
    const { workspace } = fixture();
    const source = join(workspace, 'raced.md');
    writeFileSync(source, 'contents');
    const originalRealpath = fs.realpathSync;
    const realpath = spyOn(fs, 'realpathSync').mockImplementation(((path: fs.PathLike) => {
      if (path === source) throw Object.assign(new Error('denied'), { code: 'EACCES' });
      return originalRealpath(path);
    }) as typeof fs.realpathSync);

    try {
      expect(
        resolveAgentSources([{ path: source, origin: 'workspace', kind: 'file' }], {
          workspace,
        }),
      ).toEqual({
        sources: [],
        diagnostics: [
          {
            code: 'source-unreadable',
            severity: 'error',
            path: source,
            origin: 'workspace',
            precedence: 0,
            message: `Source is unreadable: ${source}`,
          },
        ],
      });
    } finally {
      realpath.mockRestore();
    }
  });
});
