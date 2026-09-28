import { describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { ScriptedChatModel, toolCall, toolCallResponse } from '@di-framework/ai';
import {
  askUserQuestionTool,
  createSkillsToolbox,
  listDirectoryTool,
  memoryTools,
  resolveSkillPackageDirectories,
  SkillsAgent,
  skillsToolbox,
  skillsToolboxAsMcp,
  taskTool,
  todoWriteTool,
} from '../src/index.ts';

function writeSkill(
  root: string,
  name: string,
  extras?: { allowedTools?: string; body?: string },
): string {
  const dir = join(root, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, 'SKILL.md'),
    `---
name: ${name}
description: Use when reviewing ${name} files for review and audit.
${extras?.allowedTools ? `allowed-tools: ${extras.allowedTools}\n` : ''}---

${extras?.body ?? `# ${name}`}
`,
  );
  return dir;
}

describe('listDirectoryTool', () => {
  test('lists files inside the sandbox', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-ls-'));
    writeFileSync(join(root, 'a.txt'), 'a');
    mkdirSync(join(root, 'sub'));
    const tool = listDirectoryTool({ allowedDirectories: [root], workingDirectory: root });
    const out = await tool.call(JSON.stringify({}));
    expect(out).toContain('a.txt');
    expect(out).toContain('sub/');
  });
});

describe('todoWriteTool', () => {
  test('renders progress and rejects two in_progress items', async () => {
    const tool = todoWriteTool();
    const ok = await tool.call(
      JSON.stringify({
        todos: [
          { content: 'One', status: 'completed' },
          { content: 'Two', status: 'in_progress' },
        ],
      }),
    );
    expect(ok).toContain('1/2');
    expect(ok).toContain('[→] Two');
    const bad = await tool.call(
      JSON.stringify({
        todos: [
          { content: 'A', status: 'in_progress' },
          { content: 'B', status: 'in_progress' },
        ],
      }),
    );
    expect(bad).toContain('exactly one');
  });
});

describe('askUserQuestionTool', () => {
  test('delegates to the handler', async () => {
    const tool = askUserQuestionTool({
      questionHandler: (questions) => ({ [questions[0]?.question ?? 'q']: 'Day.js' }),
    });
    const out = await tool.call(
      JSON.stringify({
        questions: [{ question: 'Which library?', header: 'Lib' }],
      }),
    );
    expect(out).toContain('Day.js');
  });
});

describe('memoryTools', () => {
  test('writes, views, edits, renames, and deletes', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-mem-'));
    const [view, write, edit, del, rename] = memoryTools({ directory: root });
    if (!view || !write || !edit || !del || !rename) throw new Error('missing memory tools');
    expect(await write.call(JSON.stringify({ path: 'user.md', content: 'prefers bun' }))).toContain(
      'Wrote',
    );
    expect(await view.call(JSON.stringify({ path: 'user.md' }))).toContain('prefers bun');
    expect(
      await edit.call(
        JSON.stringify({ path: 'user.md', oldString: 'bun', newString: 'TypeScript' }),
      ),
    ).toContain('Updated');
    expect(await rename.call(JSON.stringify({ from: 'user.md', to: 'prefs.md' }))).toContain(
      'Renamed',
    );
    expect(await del.call(JSON.stringify({ path: 'prefs.md' }))).toContain('Deleted');
  });
});

describe('allowed-tools and per-skill sandbox', () => {
  test('blocks tools not listed after Skill activate', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-allow-'));
    writeSkill(root, 'code-reviewer', { allowedTools: 'Read, Grep' });
    const tools = skillsToolbox({
      directories: [root],
      workspace: root,
      write: true,
    });
    const skill = tools.find((t) => t.toolDefinition.name === 'Skill');
    const write = tools.find((t) => t.toolDefinition.name === 'Write');
    if (!skill || !write) throw new Error('missing tools');
    expect(
      await write.call(JSON.stringify({ filePath: join(root, 'x.txt'), content: 'ok' })),
    ).toContain('created');
    await skill.call(JSON.stringify({ command: 'code-reviewer' }));
    expect(
      await write.call(JSON.stringify({ filePath: join(root, 'y.txt'), content: 'no' })),
    ).toContain('not in this skill');
  });
});

describe('packages', () => {
  test('resolves a local package skills field', () => {
    const pkg = mkdtempSync(join(tmpdir(), 'ai-utils-pkg-'));
    mkdirSync(join(pkg, 'skills', 'xlsx'), { recursive: true });
    writeFileSync(
      join(pkg, 'package.json'),
      JSON.stringify({ name: 'demo-skills', skills: './skills' }),
    );
    writeFileSync(
      join(pkg, 'skills', 'xlsx', 'SKILL.md'),
      `---
name: xlsx
description: Build spreadsheets when asked for a spreadsheet.
---
# x
`,
    );
    const dirs = resolveSkillPackageDirectories([pkg]);
    expect(dirs.some((dir) => dir.endsWith('skills'))).toBe(true);
    const box = createSkillsToolbox({
      directories: [],
      packages: [pkg],
      workspace: pkg,
      sourceMode: 'replace',
    });
    expect(box.skills.map((s) => s.name)).toEqual(['xlsx']);
  });
});

describe('taskTool', () => {
  test('runs a nested agent and hides AskUserQuestion', async () => {
    const child = new ScriptedChatModel([
      {
        respond: (prompt) => {
          expect(JSON.stringify(prompt)).not.toContain('AskUserQuestion');
          return 'sub done';
        },
      },
    ]);
    const tool = taskTool({
      chatModel: child,
      tools: [askUserQuestionTool({ questionHandler: () => ({}) })],
    });
    expect(await tool.call(JSON.stringify({ prompt: 'summarize' }))).toBe('sub done');
  });
});

describe('skillsToolboxAsMcp', () => {
  test('wraps toolbox callbacks', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-mcp-'));
    writeSkill(root, 'code-reviewer');
    const mcp = skillsToolboxAsMcp({ directories: [root], workspace: root, todos: false });
    expect(mcp.some((entry) => entry.descriptor.name === 'Skill')).toBe(true);
  });
});

describe('createSkillsAgent', () => {
  test('runs Skill through the assembled agent', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-agent-'));
    writeSkill(root, 'code-reviewer', { body: '# reviewer\nBe thorough.' });
    const model = new ScriptedChatModel([
      { respond: toolCallResponse([toolCall('c1', 'Skill', { command: 'code-reviewer' })]) },
      {
        respond: (prompt) => {
          expect(JSON.stringify(prompt.messages)).toContain('Be thorough');
          return 'done';
        },
      },
    ]);
    const agent = SkillsAgent.builder()
      .chatModel(model)
      .addSkillsDirectory(root)
      .workspace(root)
      .todos(false)
      .build();
    const { content } = await agent.chat('review');
    expect(content).toBe('done');
  });
});
