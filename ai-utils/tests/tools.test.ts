import { describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { bashTool, editTool, globTool, grepTool, readTool, writeTool } from '../src/index.ts';

describe('readTool', () => {
  test('returns numbered lines and paginates', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-read-'));
    const file = join(root, 'notes.md');
    writeFileSync(file, ['alpha', 'bravo', 'charlie', 'delta'].join('\n'));
    const tool = readTool({ allowedDirectories: [root] });

    const all = await tool.call(JSON.stringify({ filePath: file }));
    expect(all).toContain('Showing lines 1-4 of 4');
    expect(all).toContain('     2\tbravo');

    const page = await tool.call(JSON.stringify({ filePath: file, offset: 3, limit: 1 }));
    expect(page).toContain('Showing lines 3-3 of 4');
    expect(page).toContain('charlie');
    expect(page).not.toContain('delta');
  });

  test('truncates long lines and reports empty / missing / directory', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-read-'));
    const long = join(root, 'long.txt');
    writeFileSync(long, `${'x'.repeat(50)}y`);
    const empty = join(root, 'empty.txt');
    writeFileSync(empty, '');
    const tool = readTool({ allowedDirectories: [root], maxLineChars: 10 });

    const truncated = await tool.call(JSON.stringify({ filePath: long }));
    expect(truncated).toContain('... (line truncated)');
    expect(await tool.call(JSON.stringify({ filePath: empty }))).toContain('File is empty');
    expect(await tool.call(JSON.stringify({ filePath: join(root, 'noop') }))).toContain(
      'does not exist',
    );
    expect(await tool.call(JSON.stringify({ filePath: root }))).toContain('directory');
  });

  test('denies a path outside the sandbox', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-read-'));
    const tool = readTool({ allowedDirectories: [root] });
    const result = await tool.call(JSON.stringify({ filePath: '/etc/passwd' }));
    expect(result).toContain('outside the allowed directories');
  });
});

describe('globTool', () => {
  test('finds files by pattern under the search root', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-glob-'));
    mkdirSync(join(root, 'references'));
    writeFileSync(join(root, 'references', 'checklist.md'), '# list');
    writeFileSync(join(root, 'SKILL.md'), 'x');
    const tool = globTool({ allowedDirectories: [root], workingDirectory: root });
    const md = await tool.call(JSON.stringify({ pattern: '**/*.md' }));
    expect(md).toContain('checklist.md');
    expect(md).toContain('SKILL.md');
    const none = await tool.call(JSON.stringify({ pattern: '**/*.py' }));
    expect(none).toContain('No files matched');
  });

  test('rejects a search path outside allowed roots', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-glob-'));
    const tool = globTool({ allowedDirectories: [root] });
    const result = await tool.call(JSON.stringify({ pattern: '*', path: '/tmp' }));
    expect(result).toContain('outside the allowed directories');
  });
});

describe('grepTool', () => {
  test('finds content, files, and counts', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-grep-'));
    mkdirSync(join(root, 'src'));
    writeFileSync(join(root, 'src', 'a.ts'), 'const TODO = 1;\nconst ok = 2;\n');
    writeFileSync(join(root, 'src', 'b.md'), 'TODO later\n');
    writeFileSync(join(root, 'src', 'c.ts'), 'nothing here\n');
    const tool = grepTool({ allowedDirectories: [root], workingDirectory: root });

    const content = await tool.call(JSON.stringify({ pattern: 'TODO', glob: '**/*.ts' }));
    expect(content).toContain('a.ts:1:');
    expect(content).not.toContain('b.md');

    const files = await tool.call(
      JSON.stringify({ pattern: 'TODO', outputMode: 'files_with_matches' }),
    );
    expect(files).toContain('a.ts');
    expect(files).toContain('b.md');

    const counts = await tool.call(JSON.stringify({ pattern: 'TODO', outputMode: 'count' }));
    expect(counts).toMatch(/a\.ts:1/);
  });

  test('rejects a search path outside allowed roots', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-grep-'));
    const tool = grepTool({ allowedDirectories: [root] });
    const result = await tool.call(JSON.stringify({ pattern: 'x', path: '/tmp' }));
    expect(result).toContain('outside the allowed directories');
  });
});

describe('writeTool and editTool', () => {
  test('creates, overwrites, and edits inside the sandbox', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-write-'));
    const writer = writeTool({ allowedDirectories: [root] });
    const editor = editTool({ allowedDirectories: [root] });
    const file = join(root, 'nested', 'note.txt');

    const created = await writer.call(JSON.stringify({ filePath: file, content: 'hello world' }));
    expect(created).toContain('created');
    const overwritten = await writer.call(
      JSON.stringify({ filePath: file, content: 'hello world' }),
    );
    expect(overwritten).toContain('overwrote');

    const edited = await editor.call(
      JSON.stringify({ filePath: file, oldString: 'world', newString: 'there' }),
    );
    expect(edited).toContain('has been updated');
    expect(await Bun.file(file).text()).toBe('hello there');

    await writer.call(JSON.stringify({ filePath: file, content: 'aa aa' }));
    const ambiguous = await editor.call(
      JSON.stringify({ filePath: file, oldString: 'aa', newString: 'bb' }),
    );
    expect(ambiguous).toContain('appears 2 times');
    const all = await editor.call(
      JSON.stringify({ filePath: file, oldString: 'aa', newString: 'bb', replaceAll: true }),
    );
    expect(all).toContain('has been updated');
    expect(await Bun.file(file).text()).toBe('bb bb');
  });

  test('denies writes outside the sandbox', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-write-'));
    const writer = writeTool({ allowedDirectories: [root] });
    const result = await writer.call(
      JSON.stringify({ filePath: '/tmp/outside-ai-utils.txt', content: 'noop' }),
    );
    expect(result).toContain('outside the allowed directories');
  });
});

describe('bashTool', () => {
  test('runs a command inside the allowed cwd', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-bash-'));
    writeFileSync(join(root, 'hello.txt'), 'hi');
    const tool = bashTool({ allowedDirectories: [root], workingDirectory: root });
    const out = await tool.call(JSON.stringify({ command: 'cat hello.txt' }));
    expect(out).toContain('exit: 0');
    expect(out).toContain('hi');
  });

  test('HITL confirm can reject a command', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-bash-'));
    const tool = bashTool({
      allowedDirectories: [root],
      workingDirectory: root,
      confirm: () => false,
    });
    const out = await tool.call(JSON.stringify({ command: 'echo hi' }));
    expect(out).toContain('not approved');
  });

  test('rejects a cwd outside the sandbox', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-bash-'));
    const tool = bashTool({ allowedDirectories: [root], workingDirectory: root });
    const out = await tool.call(JSON.stringify({ command: 'pwd', cwd: '/tmp' }));
    expect(out).toContain('outside the allowed directories');
  });

  test('times out a long-running command', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-bash-'));
    const tool = bashTool({
      allowedDirectories: [root],
      workingDirectory: root,
      timeoutMs: 50,
    });
    const out = await tool.call(JSON.stringify({ command: 'sleep 5', timeout: 80 }));
    expect(out).toContain('timed out');
  });
});
