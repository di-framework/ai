import { describe, expect, test } from 'bun:test';
import { existsSync, mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { compileAiIgnorePolicy, loadAiIgnorePolicy } from '../src/policy/index.ts';
import { skillsToolboxBuilderFrom } from '../src/skills/skills-decorators.ts';
import { SkillsToolbox } from '../src/skills/skills-toolbox.ts';
import type { AiIgnoreEnforcement, AiIgnoreToolPolicy } from '../src/tools/aiignore-enforcement.ts';
import { editTool } from '../src/tools/edit-tool.ts';
import { listDirectoryTool } from '../src/tools/list-directory-tool.ts';
import { readTool } from '../src/tools/read-tool.ts';
import { writeTool } from '../src/tools/write-tool.ts';

const modes = ['discovery', 'read', 'read-write'] as const satisfies readonly AiIgnoreEnforcement[];

function fixture(pattern = '*.ignored'): {
  root: string;
  workspace: string;
  policy: AiIgnoreToolPolicy['policy'];
} {
  const root = mkdtempSync(join(tmpdir(), 'ai-utils-aiignore-tools-'));
  const workspace = join(root, 'workspace');
  mkdirSync(workspace);
  const policy = compileAiIgnorePolicy(pattern, { workspace });
  return { root, workspace, policy };
}

function enforcement(
  policy: AiIgnoreToolPolicy['policy'],
  mode: AiIgnoreEnforcement,
): AiIgnoreToolPolicy {
  return { policy, enforcement: mode };
}

function expectPolicyRejection(
  result: unknown,
  path: string,
  policyPath: string,
  hiddenContent: string,
): void {
  const output = String(result);
  expect(output).toContain(path);
  expect(output).toContain(policyPath);
  expect(output).toContain('rule line 1');
  expect(output).not.toContain(hiddenContent);
  expect(output).not.toContain('*.ignored');
}

describe('Read .aiignore enforcement', () => {
  for (const mode of modes) {
    test(`${mode} has documented Read semantics`, async () => {
      const { workspace, policy } = fixture();
      const file = join(workspace, `${mode}.ignored`);
      const secret = `secret-${mode}`;
      writeFileSync(file, secret);
      const tool = readTool({
        allowedDirectories: [workspace],
        aiIgnore: enforcement(policy, mode),
      });

      const result = await tool.call(JSON.stringify({ filePath: file }));
      if (mode === 'discovery') {
        expect(result).toContain(secret);
      } else {
        expectPolicyRejection(result, file, policy.source.path, secret);
      }
    });
  }

  test('keeps the root policy readable for bootstrap evaluation', async () => {
    const { workspace } = fixture();
    const policyPath = join(workspace, '.aiignore');
    writeFileSync(policyPath, '*\n');
    const policy = loadAiIgnorePolicy({ workspace });
    const tool = readTool({
      allowedDirectories: [workspace],
      aiIgnore: enforcement(policy, 'read-write'),
    });

    expect(await tool.call(JSON.stringify({ filePath: policyPath }))).toContain('*');
  });
});

describe('Write .aiignore enforcement', () => {
  for (const mode of modes) {
    test(`${mode} has documented Write semantics`, async () => {
      const { workspace, policy } = fixture();
      const file = join(workspace, `${mode}.ignored`);
      const secret = `write-${mode}`;
      const tool = writeTool({
        allowedDirectories: [workspace],
        aiIgnore: enforcement(policy, mode),
      });

      const result = await tool.call(JSON.stringify({ filePath: file, content: secret }));
      if (mode === 'read-write') {
        expectPolicyRejection(result, file, policy.source.path, secret);
        expect(existsSync(file)).toBe(false);
      } else {
        expect(result).toContain('created');
        expect(await Bun.file(file).text()).toBe(secret);
      }
    });
  }
});

describe('Edit .aiignore enforcement', () => {
  for (const mode of modes) {
    test(`${mode} has documented Edit semantics`, async () => {
      const { workspace, policy } = fixture();
      const file = join(workspace, `${mode}.ignored`);
      const secret = `before-${mode}`;
      writeFileSync(file, secret);
      const tool = editTool({
        allowedDirectories: [workspace],
        aiIgnore: enforcement(policy, mode),
      });

      const result = await tool.call(
        JSON.stringify({ filePath: file, oldString: secret, newString: `after-${mode}` }),
      );
      if (mode === 'discovery') {
        expect(result).toContain('updated');
        expect(await Bun.file(file).text()).toBe(`after-${mode}`);
      } else {
        expectPolicyRejection(result, file, policy.source.path, secret);
        expect(await Bun.file(file).text()).toBe(secret);
      }
    });
  }
});

describe('ListDirectory .aiignore enforcement', () => {
  for (const mode of modes) {
    test(`${mode} filters entries and rejects an ignored listing root`, async () => {
      const { workspace } = fixture();
      writeFileSync(join(workspace, '.aiignore'), '*.ignored\n');
      const ignoredFile = join(workspace, `${mode}.ignored`);
      const ignoredDirectory = join(workspace, 'private.ignored');
      const visible = join(workspace, 'visible.txt');
      writeFileSync(ignoredFile, `hidden-${mode}`);
      mkdirSync(ignoredDirectory);
      writeFileSync(join(ignoredDirectory, 'contents.txt'), 'hidden directory content');
      writeFileSync(visible, 'visible');
      const policy = loadAiIgnorePolicy({ workspace });
      const tool = listDirectoryTool({
        allowedDirectories: [workspace],
        workingDirectory: workspace,
        aiIgnore: enforcement(policy, mode),
      });

      const listing = await tool.call(JSON.stringify({}));
      expect(listing).toContain(visible);
      expect(listing).toContain(join(workspace, '.aiignore'));
      expect(listing).not.toContain(ignoredFile);
      expect(listing).not.toContain(ignoredDirectory);

      const rejected = await tool.call(JSON.stringify({ path: ignoredDirectory }));
      expectPolicyRejection(
        rejected,
        ignoredDirectory,
        policy.source.path,
        'hidden directory content',
      );
    });
  }
});

describe('.aiignore and stronger sandbox policy', () => {
  test('negation cannot re-enable sandbox-denied paths for any direct tool', async () => {
    const { root, workspace, policy } = fixture('!*');
    const outsideFile = join(root, 'outside.txt');
    const outsideDirectory = join(root, 'outside-directory');
    writeFileSync(outsideFile, 'outside secret');
    mkdirSync(outsideDirectory);
    const fileLink = join(workspace, 'file-link');
    const directoryLink = join(workspace, 'directory-link');
    symlinkSync(outsideFile, fileLink);
    symlinkSync(outsideDirectory, directoryLink, 'dir');
    const aiIgnore = enforcement(policy, 'read-write');

    const results = await Promise.all([
      readTool({ allowedDirectories: [workspace], aiIgnore }).call(
        JSON.stringify({ filePath: fileLink }),
      ),
      writeTool({ allowedDirectories: [workspace], aiIgnore }).call(
        JSON.stringify({ filePath: fileLink, content: 'overwrite' }),
      ),
      editTool({ allowedDirectories: [workspace], aiIgnore }).call(
        JSON.stringify({ filePath: fileLink, oldString: 'outside', newString: 'changed' }),
      ),
      listDirectoryTool({ allowedDirectories: [workspace], aiIgnore }).call(
        JSON.stringify({ path: directoryLink }),
      ),
    ]);

    for (const result of results) {
      expect(result).toContain('outside the allowed directories');
      expect(result).not.toContain('.aiignore policy');
    }
    expect(await Bun.file(outsideFile).text()).toBe('outside secret');
  });

  test('omits sandbox-denied children before applying discovery policy', async () => {
    const { root, workspace, policy } = fixture('!link');
    const outside = join(root, 'outside.txt');
    const link = join(workspace, 'link');
    writeFileSync(outside, 'secret');
    symlinkSync(outside, link);
    const tool = listDirectoryTool({
      allowedDirectories: [workspace],
      workingDirectory: workspace,
      aiIgnore: enforcement(policy, 'discovery'),
    });

    expect(await tool.call(JSON.stringify({}))).not.toContain(link);
  });

  test('does not apply a workspace policy to another explicitly allowed root', async () => {
    const { root, workspace, policy } = fixture('*');
    const extra = join(root, 'extra');
    const file = join(extra, 'shared.txt');
    mkdirSync(extra);
    writeFileSync(file, 'shared content');
    const tool = readTool({
      allowedDirectories: [workspace, extra],
      aiIgnore: enforcement(policy, 'read'),
    });

    expect(await tool.call(JSON.stringify({ filePath: file }))).toContain('shared content');
  });

  test('preserves the policy workspace sandbox inside a broader allowed root', async () => {
    const { root, workspace, policy } = fixture('!link');
    const outside = join(root, 'outside.txt');
    const link = join(workspace, 'link');
    writeFileSync(outside, 'secret');
    symlinkSync(outside, link);
    const tool = readTool({
      allowedDirectories: [root],
      aiIgnore: enforcement(policy, 'read'),
    });

    const result = await tool.call(JSON.stringify({ filePath: link }));
    expect(result).toContain('outside the allowed directories');
    expect(result).not.toContain('.aiignore policy');
  });
});

describe('SkillsToolbox .aiignore integration', () => {
  test('loads and applies the root policy through the fluent mode', async () => {
    const { workspace } = fixture();
    const skillDirectory = join(workspace, 'skills', 'reviewer');
    const ignored = join(workspace, 'private.txt');
    mkdirSync(skillDirectory, { recursive: true });
    writeFileSync(
      join(skillDirectory, 'SKILL.md'),
      '---\nname: reviewer\ndescription: Review project files when asked to inspect code.\n---\n# Reviewer\n',
    );
    writeFileSync(join(workspace, '.aiignore'), 'private.txt\n');
    writeFileSync(ignored, 'private content');

    const tools = SkillsToolbox.builder()
      .workspace(workspace)
      .addSkillsDirectory(skillDirectory)
      .aiIgnore('read-write')
      .write()
      .glob(false)
      .grep(false)
      .todos(false)
      .buildTools();
    const read = tools.find((tool) => tool.toolDefinition.name === 'Read');
    const write = tools.find((tool) => tool.toolDefinition.name === 'Write');
    if (read == null || write == null) throw new Error('missing direct file tools');

    expect(await read.call(JSON.stringify({ filePath: ignored }))).toContain('.aiignore policy');
    expect(await write.call(JSON.stringify({ filePath: ignored, content: 'changed' }))).toContain(
      '.aiignore policy',
    );
    expect(await Bun.file(ignored).text()).toBe('private content');
  });

  test('preserves an enforcement override when pre-filling a toolbox builder', () => {
    class ConfiguredAgent {}

    expect(
      skillsToolboxBuilderFrom(ConfiguredAgent, { aiIgnore: 'read' }).toOptions().aiIgnore,
    ).toBe('read');
  });
});
