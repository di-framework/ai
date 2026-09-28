import { describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  agentSkill,
  collectSkills,
  DEFAULT_SKILL_TOOL_NAME,
  formatSkillLoadResult,
  formatSkillNotFound,
  loadSkillFile,
  loadSkillsDirectories,
  loadSkillsDirectory,
  parseSkillMarkdown,
  SkillsTool,
  skillsTool,
  skillToXml,
} from '../src/index.ts';

function writeSkill(
  root: string,
  folder: string,
  body: string,
  extraFiles?: Record<string, string>,
): string {
  const dir = join(root, folder);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'SKILL.md'), body);
  for (const [rel, content] of Object.entries(extraFiles ?? {})) {
    const path = join(dir, rel);
    mkdirSync(join(path, '..'), { recursive: true });
    writeFileSync(path, content);
  }
  return dir;
}

describe('parseSkillMarkdown', () => {
  test('extracts front matter and body', () => {
    const markdown = `---
name: code-reviewer
description: Reviews Java code
---

# Code Reviewer

Check for nulls.
`;
    const skill = parseSkillMarkdown(markdown, { basePath: '/tmp/code-reviewer' });
    expect(skill.name).toBe('code-reviewer');
    expect(skill.description).toBe('Reviews Java code');
    expect(skill.basePath).toBe('/tmp/code-reviewer');
    expect(skill.content).toContain('# Code Reviewer');
    expect(skill.frontMatter.name).toBe('code-reviewer');
    expect(skill.source).toBe(markdown);
  });

  test('strips surrounding quotes and keeps values after the first colon', () => {
    const skill = parseSkillMarkdown(
      `---
name: "quoted"
description: 'Use when: reviewing'
note: a:b:c
---

body
`,
      { fallbackName: 'ignored' },
    );
    expect(skill.name).toBe('quoted');
    expect(skill.description).toBe('Use when: reviewing');
    expect(skill.frontMatter.note).toBe('a:b:c');
  });

  test('parses allowed-tools as a comma list', () => {
    const skill = parseSkillMarkdown(
      `---
name: code-reviewer
description: Reviews code when asked to review.
allowed-tools: Read, Grep, Bash
---

body
`,
    );
    expect(skill.allowedTools).toEqual(['Read', 'Grep', 'Bash']);
  });

  test('treats a document without a closing delimiter as body', () => {
    const skill = parseSkillMarkdown('---\nname: noop\nno close', { fallbackName: 'folder' });
    expect(skill.name).toBe('folder');
    expect(skill.content).toContain('name: noop');
  });

  test('uses the folder name when front matter has no name', () => {
    const skill = parseSkillMarkdown('# just body', { fallbackName: 'from-folder' });
    expect(skill.name).toBe('from-folder');
    expect(skill.content).toBe('# just body');
  });

  test('throws when neither front matter nor fallback supplies a name', () => {
    expect(() => parseSkillMarkdown('# no name')).toThrow(/missing a name/);
  });

  test('skips blank and colon-less front matter lines', () => {
    const skill = parseSkillMarkdown(
      `---
name: keep

not-a-pair
:orphan
---
ok
`,
    );
    expect(skill.name).toBe('keep');
    expect(skill.content).toBe('ok');
    expect(skill.frontMatter['not-a-pair']).toBeUndefined();
  });
});

describe('agentSkill', () => {
  test('builds an in-memory skill', () => {
    const skill = agentSkill({
      name: 'tutor',
      description: 'Explain concepts',
      content: 'Be patient.',
      basePath: '/skills/tutor',
    });
    expect(skill.frontMatter.name).toBe('tutor');
    expect(skill.frontMatter.description).toBe('Explain concepts');
  });

  test('rejects a blank name', () => {
    expect(() => agentSkill({ name: '  ', content: 'x' })).toThrow('Skill name is required');
  });
});

describe('loadSkillsDirectory', () => {
  test('walks nested SKILL.md files and skips node_modules', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-skills-'));
    writeSkill(
      root,
      'reviewer',
      `---
name: reviewer
description: Review code
---
Review it.
`,
    );
    writeSkill(
      join(root, 'nested'),
      'research',
      `---
name: research
---
Search first.
`,
    );
    writeSkill(
      join(root, 'node_modules', 'pkg'),
      'ignored',
      `---
name: ignored
---
noop
`,
    );

    const skills = loadSkillsDirectory(root);
    const names = skills.map((s) => s.name).sort();
    expect(names).toEqual(['research', 'reviewer']);
    expect(skills.find((s) => s.name === 'reviewer')?.content).toContain('Review it.');
  });

  test('falls back to the folder name when SKILL.md has no name', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-skills-'));
    writeSkill(root, 'pdf', '# PDF help');
    const [skill] = loadSkillsDirectory(root);
    expect(skill?.name).toBe('pdf');
  });

  test('loadSkillsDirectories concatenates roots', () => {
    const a = mkdtempSync(join(tmpdir(), 'ai-utils-a-'));
    const b = mkdtempSync(join(tmpdir(), 'ai-utils-b-'));
    writeSkill(a, 'one', '---\nname: one\n---\nA');
    writeSkill(b, 'two', '---\nname: two\n---\nB');
    expect(
      loadSkillsDirectories([a, b])
        .map((s) => s.name)
        .sort(),
    ).toEqual(['one', 'two']);
  });

  test('throws when the path is missing or not a directory', () => {
    expect(() => loadSkillsDirectory(join(tmpdir(), 'does-not-exist-ai-utils'))).toThrow(
      /does not exist/,
    );
    const file = join(mkdtempSync(join(tmpdir(), 'ai-utils-file-')), 'SKILL.md');
    writeFileSync(file, '---\nname: x\n---\n');
    expect(() => loadSkillsDirectory(file)).toThrow(/not a directory/);
  });

  test('loadSkillFile reads one SKILL.md', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-file-'));
    const dir = writeSkill(root, 'xlsx', '---\nname: xlsx\n---\nSheets');
    const skill = loadSkillFile(join(dir, 'SKILL.md'));
    expect(skill.name).toBe('xlsx');
    expect(skill.basePath).toBe(dir);
  });

  test('loadSkillFile throws for missing or non-file paths', () => {
    expect(() => loadSkillFile(join(tmpdir(), 'missing-SKILL.md'))).toThrow(/does not exist/);
    const dir = mkdtempSync(join(tmpdir(), 'ai-utils-dir-'));
    expect(() => loadSkillFile(dir)).toThrow(/not a file/);
  });
});

describe('skillsTool', () => {
  test('embeds available skills and returns the body on invoke', async () => {
    const skill = agentSkill({
      name: 'xlsx',
      description: 'Make spreadsheets',
      content: 'Use tables.',
      basePath: '/tmp/xlsx',
    });
    const tool = skillsTool({ skills: [skill] });
    expect(tool.toolDefinition.name).toBe(DEFAULT_SKILL_TOOL_NAME);
    expect(tool.toolDefinition.description).toContain('<name>xlsx</name>');
    expect(tool.toolDefinition.description).toContain('Make spreadsheets');

    const loaded = await tool.call(JSON.stringify({ command: 'xlsx' }));
    expect(loaded).toBe(formatSkillLoadResult(skill));
  });

  test('returns Skill not found for unknown or blank names', async () => {
    const tool = skillsTool({
      skills: [agentSkill({ name: 'pdf', content: 'PDF' })],
    });
    expect(await tool.call(JSON.stringify({ command: 'docx' }))).toBe(formatSkillNotFound('docx'));
    expect(await tool.call(JSON.stringify({ command: '  ' }))).toBe(formatSkillNotFound(''));
    expect(await tool.call('{}')).toBe(formatSkillNotFound(''));
  });

  test('first skill with the same name wins', async () => {
    const tool = skillsTool({
      skills: [
        agentSkill({ name: 'pdf', content: 'v1' }),
        agentSkill({ name: 'pdf', content: 'v2' }),
      ],
    });
    expect(await tool.call(JSON.stringify({ command: 'pdf' }))).toContain('v1');
  });

  test('escapes XML special characters in the tool description', () => {
    const tool = skillsTool({
      skills: [
        agentSkill({
          name: 'amp',
          description: 'A & B <C> "quote"',
          content: 'x',
        }),
      ],
    });
    expect(tool.toolDefinition.description).toContain('A &amp; B &lt;C&gt; &quot;quote&quot;');
  });

  test('throws when no skills are configured', () => {
    expect(() => skillsTool({})).toThrow(/At least one skill/);
  });

  test('loads from directories and files', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-tool-'));
    writeSkill(root, 'dir-skill', '---\nname: from-dir\n---\nDir body');
    const fileDir = writeSkill(root, 'file-skill', '---\nname: from-file\n---\nFile body');
    const tool = skillsTool({
      directories: [join(root, 'dir-skill')],
      files: [join(fileDir, 'SKILL.md')],
    });
    expect(await tool.call(JSON.stringify({ command: 'from-dir' }))).toContain('Dir body');
    expect(await tool.call(JSON.stringify({ command: 'from-file' }))).toContain('File body');
  });

  test('SkillsTool.builder and SkillsTool.of match skillsTool()', async () => {
    const skill = agentSkill({ name: 'one', content: 'body' });
    const viaOf = SkillsTool.of({ skills: [skill] });
    const viaBuilder = SkillsTool.builder()
      .addSkill(skill)
      .addSkills([])
      .toolName('Skill')
      .toolDescriptionTemplate('Available:\n%s')
      .build();
    expect(viaOf.toolDefinition.name).toBe('Skill');
    expect(viaBuilder.toolDefinition.description).toContain('<name>one</name>');
    expect(viaBuilder.toolDefinition.description.startsWith('Available:')).toBe(true);
    expect(await viaBuilder.call(JSON.stringify({ command: 'one' }))).toContain('body');
  });

  test('template without %s appends the skill xml', () => {
    const tool = skillsTool({
      skills: [agentSkill({ name: 'x', content: 'y' })],
      toolDescriptionTemplate: 'plain',
    });
    expect(tool.toolDefinition.description).toContain('plain');
    expect(tool.toolDefinition.description).toContain('<name>x</name>');
  });

  test('builder addSkillsDirectory / addSkillsFile / addSkillsDirectories', async () => {
    const a = mkdtempSync(join(tmpdir(), 'ai-utils-ba-'));
    const b = mkdtempSync(join(tmpdir(), 'ai-utils-bb-'));
    writeSkill(a, 'alpha', '---\nname: alpha\n---\nA');
    writeSkill(b, 'beta', '---\nname: beta\n---\nB');
    const extra = writeSkill(a, 'gamma', '---\nname: gamma\n---\nC');
    const tool = SkillsTool.builder()
      .addSkillsDirectory(a)
      .addSkillsDirectories([b])
      .addSkillsFile(join(extra, 'SKILL.md'))
      .build();
    expect(await tool.call(JSON.stringify({ command: 'alpha' }))).toContain('A');
    expect(await tool.call(JSON.stringify({ command: 'beta' }))).toContain('B');
    expect(await tool.call(JSON.stringify({ command: 'gamma' }))).toContain('C');
  });

  test('collectSkills and skillToXml are exported helpers', () => {
    const skill = agentSkill({ name: 'n', description: 'd', content: 'c' });
    expect(collectSkills({ skills: [skill] })).toHaveLength(1);
    expect(collectSkills({ directories: [] })).toHaveLength(0);
    expect(skillToXml(skill)).toContain('<description>d</description>');
  });
});
