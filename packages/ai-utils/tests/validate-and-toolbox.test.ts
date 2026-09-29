import { describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { ChatAgent, ScriptedChatModel, toolCall, toolCallResponse } from '@di-framework/ai';
import {
  agentSkill,
  createSkillsToolbox,
  SkillsAgent,
  SkillsToolbox,
  skillsToolbox,
  validateSkill,
  validateSkillName,
} from '../src/index.ts';

function writeValidSkill(root: string, name: string, extra?: { reference?: string }): string {
  const dir = join(root, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, 'SKILL.md'),
    `---
name: ${name}
description: Use when reviewing ${name} files. Trigger words include review and audit.
---

# ${name}

Read references/checklist.md for the checklist.
`,
  );
  if (extra?.reference) {
    mkdirSync(join(dir, 'references'));
    writeFileSync(join(dir, 'references', 'checklist.md'), extra.reference);
  }
  return dir;
}

describe('validateSkill', () => {
  test('accepts a spec-compliant name and description', () => {
    expect(() =>
      validateSkill(
        agentSkill({
          name: 'code-reviewer',
          description: 'Reviews TypeScript. Use when the user asks to review code.',
          content: 'Be thorough.',
          basePath: '.',
        }),
        { matchDirectoryName: false },
      ),
    ).not.toThrow();
  });

  test('rejects invalid names and missing descriptions', () => {
    expect(validateSkillName('PDF-Processing')).toBeDefined();
    expect(validateSkillName('-pdf')).toBeDefined();
    expect(validateSkillName('pdf--processing')).toBeDefined();
    expect(validateSkillName('a'.repeat(65))).toBeDefined();
    expect(() =>
      validateSkill(agentSkill({ name: 'ok', content: 'x' }), { matchDirectoryName: false }),
    ).toThrow(/description is required/);
  });

  test('requires the directory basename to match the name', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-val-'));
    const dir = join(root, 'other-name');
    mkdirSync(dir);
    expect(() =>
      validateSkill(
        agentSkill({
          name: 'code-reviewer',
          description: 'Reviews code when asked to review.',
          content: 'x',
          basePath: dir,
        }),
      ),
    ).toThrow(/must match the skill directory name/);
  });
});

describe('skillsToolbox', () => {
  test('builds Skill, Read, Glob, Grep, and optional Write/Edit/Bash', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-box-'));
    writeValidSkill(root, 'code-reviewer');
    const withoutShell = skillsToolbox({
      directories: [root],
      workspace: root,
    });
    expect(withoutShell.map((t) => t.toolDefinition.name)).toEqual([
      'Skill',
      'Read',
      'ListDirectory',
      'Glob',
      'Grep',
      'TodoWrite',
    ]);

    const withMutators = skillsToolbox({
      directories: [root],
      workspace: root,
      write: true,
      shell: true,
    });
    expect(withMutators.map((t) => t.toolDefinition.name)).toEqual([
      'Skill',
      'Read',
      'ListDirectory',
      'Glob',
      'Grep',
      'Write',
      'Edit',
      'Bash',
      'TodoWrite',
    ]);
  });

  test('fails closed when a loaded skill is invalid', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-box-'));
    const dir = join(root, 'bad-skill');
    mkdirSync(dir);
    writeFileSync(join(dir, 'SKILL.md'), '---\nname: bad-skill\n---\n# no description\n');
    expect(() => skillsToolbox({ directories: [root], workspace: root })).toThrow(
      /description is required/,
    );
  });

  test('ChatAgent can Skill then Read a reference file', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-box-'));
    const skillDir = writeValidSkill(root, 'code-reviewer', {
      reference: '- Check for nulls\n- Prefer early return',
    });
    const checklist = join(skillDir, 'references', 'checklist.md');

    const model = new ScriptedChatModel([
      {
        respond: toolCallResponse([toolCall('c1', 'Skill', { command: 'code-reviewer' })]),
      },
      {
        respond: toolCallResponse([toolCall('c2', 'Read', { filePath: checklist })]),
      },
      {
        respond: (prompt) => {
          const blob = JSON.stringify(prompt.messages);
          expect(blob).toContain('# code-reviewer');
          expect(blob).toContain('Check for nulls');
          return 'Review done using the checklist.';
        },
      },
    ]);

    const agent = ChatAgent.create({
      chatModel: model,
      tools: skillsToolbox({ directories: [root], workspace: root }),
    });
    const { content } = await agent.chat('Review this file.');
    expect(content).toBe('Review done using the checklist.');
  });

  test('replace mode keeps an in-memory catalog isolated from default directories', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-box-'));
    const box = createSkillsToolbox({
      workspace: root,
      sourceMode: 'replace',
      skills: [
        agentSkill({
          name: 'xlsx',
          description: 'Build spreadsheets when asked to make a spreadsheet.',
          content: 'Prefer a real .xlsx.',
        }),
      ],
    });
    expect(box.skills.map((s) => s.name)).toEqual(['xlsx']);
  });

  test('SkillsToolbox.builder and SkillsAgent.builder are the preferred factories', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-box-'));
    writeValidSkill(root, 'code-reviewer');
    const box = SkillsToolbox.builder()
      .addSkillsDirectory(root)
      .sourceMode('replace')
      .workspace(root)
      .write()
      .shell()
      .todos(false)
      .build();
    expect(box.tools.map((t) => t.toolDefinition.name)).toEqual([
      'Skill',
      'Read',
      'ListDirectory',
      'Glob',
      'Grep',
      'Write',
      'Edit',
      'Bash',
    ]);
    expect(
      SkillsToolbox.of({ directories: [root], workspace: root, sourceMode: 'replace' }).skills,
    ).toHaveLength(1);

    const model = new ScriptedChatModel([
      { respond: toolCallResponse([toolCall('c1', 'Skill', { command: 'code-reviewer' })]) },
      { respond: () => 'ok' },
    ]);
    const agent = SkillsAgent.builder()
      .chatModel(model)
      .addSkillsDirectory(root)
      .sourceMode('replace')
      .workspace(root)
      .todos(false)
      .build();
    expect((await agent.chat('review')).content).toBe('ok');
  });

  test('createSkillsToolbox exposes allowed directories including skill bases', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-box-'));
    const skillDir = writeValidSkill(root, 'code-reviewer');
    const box = createSkillsToolbox({
      directories: [root],
      workspace: root,
      sourceMode: 'replace',
    });
    expect(box.skills).toHaveLength(1);
    expect(box.allowedDirectories.some((d) => d === skillDir)).toBe(true);
  });
});
