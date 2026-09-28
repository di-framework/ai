import { describe, expect, spyOn, test } from 'bun:test';
import * as fs from 'node:fs';
import { mkdirSync, mkdtempSync, realpathSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  DEFAULT_AGENT_INSTRUCTIONS_FILENAME,
  DEFAULT_AGENT_INSTRUCTIONS_MAX_BYTES,
  discoverAgentInstructions,
} from '../src/index.ts';

function workspace(): string {
  return mkdtempSync(join(tmpdir(), 'ai-utils-instructions-'));
}

describe('discoverAgentInstructions', () => {
  test('loads workspace-to-cwd instructions broad-to-specific with provenance', () => {
    const root = workspace();
    const service = join(root, 'packages', 'service');
    mkdirSync(service, { recursive: true });
    writeFileSync(join(root, 'AGENTS.md'), 'root rules');
    writeFileSync(join(root, 'packages', 'AGENTS.md'), 'package rules');
    writeFileSync(join(service, 'AGENTS.md'), 'service rules');

    const result = discoverAgentInstructions({
      workspace: root,
      workingDirectory: 'packages/service',
    });

    expect(result.content).toBe('root rules\n\npackage rules\n\nservice rules');
    expect(result.bytes).toBe(Buffer.byteLength(result.content));
    expect(
      result.sources.map(({ path, filename, directory, precedence, origin, content }) => ({
        path,
        filename,
        directory,
        precedence,
        origin,
        content,
      })),
    ).toEqual([
      {
        path: join(root, 'AGENTS.md'),
        filename: 'AGENTS.md',
        directory: root,
        precedence: 0,
        origin: 'workspace',
        content: 'root rules',
      },
      {
        path: join(root, 'packages', 'AGENTS.md'),
        filename: 'AGENTS.md',
        directory: join(root, 'packages'),
        precedence: 1,
        origin: 'workspace',
        content: 'package rules',
      },
      {
        path: join(service, 'AGENTS.md'),
        filename: 'AGENTS.md',
        directory: service,
        precedence: 2,
        origin: 'workspace',
        content: 'service rules',
      },
    ]);
    expect(result.diagnostics).toEqual([]);
  });

  test('uses only AGENTS.md by default and requires explicit fallback filenames', () => {
    const root = workspace();
    writeFileSync(join(root, '.agents.md'), 'explicit fallback');

    expect(DEFAULT_AGENT_INSTRUCTIONS_FILENAME).toBe('AGENTS.md');
    expect(DEFAULT_AGENT_INSTRUCTIONS_MAX_BYTES).toBe(32 * 1024);
    const defaultResult = discoverAgentInstructions({ workspace: root, workingDirectory: root });
    expect(defaultResult.sources).toEqual([]);
    expect(defaultResult.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
      'source-missing',
    ]);

    const configured = discoverAgentInstructions({
      workspace: root,
      workingDirectory: root,
      fallbackFilenames: ['.agents.md'],
    });
    expect(configured.content).toBe('explicit fallback');
    expect(configured.sources[0]?.filename).toBe('.agents.md');
    expect(configured.diagnostics.map((diagnostic) => diagnostic.code)).toEqual(['source-missing']);

    writeFileSync(join(root, 'AGENTS.md'), 'primary');
    expect(
      discoverAgentInstructions({
        workspace: root,
        workingDirectory: root,
        fallbackFilenames: ['.agents.md'],
      }),
    ).toMatchObject({ content: 'primary', diagnostics: [] });
  });

  test('handles .agents/AGENTS.md naturally when work is below .agents', () => {
    const root = workspace();
    const working = join(root, '.agents', 'skills', 'review');
    mkdirSync(working, { recursive: true });
    writeFileSync(join(root, 'AGENTS.md'), 'workspace');
    writeFileSync(join(root, '.agents', 'AGENTS.md'), 'agent assets');

    const result = discoverAgentInstructions({ workspace: root, workingDirectory: working });
    expect(result.content).toBe('workspace\n\nagent assets');
    expect(result.sources.map((source) => source.directory)).toEqual([root, join(root, '.agents')]);
    expect(
      result.diagnostics.filter((diagnostic) => diagnostic.code === 'source-missing'),
    ).toHaveLength(2);
  });

  test('skips empty files and whole files that exceed the combined byte limit', () => {
    const root = workspace();
    const middle = join(root, 'middle');
    const working = join(middle, 'working');
    mkdirSync(working, { recursive: true });
    writeFileSync(join(root, 'AGENTS.md'), '1234');
    writeFileSync(join(middle, 'AGENTS.md'), ' \n\t');
    writeFileSync(join(working, 'AGENTS.md'), 'abcd');

    const result = discoverAgentInstructions({
      workspace: root,
      workingDirectory: working,
      maxBytes: 9,
    });
    expect(result.content).toBe('1234');
    expect(result.bytes).toBe(4);
    expect(result.sources).toHaveLength(1);
    expect(result.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
      'instructions-empty',
      'instructions-max-bytes-exceeded',
    ]);
  });

  test('never walks above the workspace or follows symlinks out of it', () => {
    const root = workspace();
    const outside = workspace();
    const outsideFile = join(outside, 'AGENTS.md');
    writeFileSync(outsideFile, 'outside');
    symlinkSync(outsideFile, join(root, 'AGENTS.md'));

    const escapedFile = discoverAgentInstructions({ workspace: root, workingDirectory: root });
    expect(escapedFile.sources).toEqual([]);
    expect(escapedFile.diagnostics[0]?.code).toBe('source-outside-boundary');

    const escapedWorking = discoverAgentInstructions({
      workspace: root,
      workingDirectory: outside,
      allowedDirectories: [outside],
    });
    expect(escapedWorking.sources).toEqual([]);
    expect(escapedWorking.diagnostics).toEqual([
      {
        code: 'instructions-invalid-working-directory',
        severity: 'error',
        path: outside,
        origin: 'workspace',
        precedence: 0,
        message: `Working directory is outside or unavailable within the workspace: ${outside}`,
      },
    ]);
  });

  test('intersects discovery with explicitly allowed directories', () => {
    const root = workspace();
    const working = join(root, 'allowed', 'work');
    mkdirSync(working, { recursive: true });
    writeFileSync(join(root, 'AGENTS.md'), 'outside sandbox');
    writeFileSync(join(root, 'allowed', 'AGENTS.md'), 'allowed rules');

    const result = discoverAgentInstructions({
      workspace: root,
      workingDirectory: working,
      allowedDirectories: [join(root, 'allowed')],
    });
    expect(result.content).toBe('allowed rules');
    expect(result.sources.map((source) => source.path)).toEqual([
      join(root, 'allowed', 'AGENTS.md'),
    ]);
    expect(result.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
      'instructions-outside-allowed-directories',
      'source-missing',
    ]);

    const deniedWorking = discoverAgentInstructions({
      workspace: root,
      workingDirectory: working,
      allowedDirectories: [join(root, 'elsewhere')],
    });
    expect(deniedWorking.diagnostics[0]?.code).toBe('instructions-outside-allowed-directories');
  });

  test('validates fallback names and byte limits', () => {
    const root = workspace();
    expect(() =>
      discoverAgentInstructions({ workspace: root, fallbackFilenames: ['../AGENTS.md'] }),
    ).toThrow(/must be a filename/);
    expect(() => discoverAgentInstructions({ workspace: root, maxBytes: -1 })).toThrow(
      /non-negative safe integer/,
    );
  });

  test('reports a file that becomes unreadable after source resolution', () => {
    const root = workspace();
    const file = join(root, 'AGENTS.md');
    writeFileSync(file, 'rules');
    const originalOpen = fs.openSync;
    const realFile = realpathSync(file);
    const open = spyOn(fs, 'openSync').mockImplementation(((path: fs.PathLike, flags: string) => {
      if (path === realFile) throw Object.assign(new Error('denied'), { code: 'EACCES' });
      return originalOpen(path, flags);
    }) as typeof fs.openSync);

    try {
      const result = discoverAgentInstructions({ workspace: root, workingDirectory: root });
      expect(result.sources).toEqual([]);
      expect(result.diagnostics[0]?.code).toBe('source-unreadable');
    } finally {
      open.mockRestore();
    }
  });
});
