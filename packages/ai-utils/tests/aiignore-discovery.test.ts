import { describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  type AiIgnoreSuppressionDiagnostic,
  collectSkills,
  createSkillsToolbox,
  discoverAgentInstructions,
  globTool,
  grepTool,
  loadAiIgnorePolicy,
  loadSkillsDirectory,
  SkillsToolbox,
} from '../src/index.ts';

function workspace(): string {
  return mkdtempSync(join(tmpdir(), 'aiignore-discovery-'));
}

function writeSkill(root: string, name: string, body = `instructions for ${name}`): string {
  const directory = join(root, name);
  mkdirSync(directory, { recursive: true });
  writeFileSync(
    join(directory, 'SKILL.md'),
    `---\nname: ${name}\ndescription: Use ${name} when testing discovery.\n---\n${body}`,
  );
  return directory;
}

describe('.aiignore discovery enforcement', () => {
  test('prunes directories before Glob and Grep traversal and suppresses ignored files', async () => {
    const root = workspace();
    const ignoredDirectory = join(root, 'ignored-dir');
    const structuralDirectory = join(root, 'node_modules');
    mkdirSync(ignoredDirectory);
    mkdirSync(structuralDirectory);
    writeFileSync(
      join(root, '.aiignore'),
      'ignored-dir/\nignored-file.txt\n!node_modules/kept.txt',
    );
    writeFileSync(join(root, 'visible.txt'), 'needle visible');
    writeFileSync(join(root, 'ignored-file.txt'), 'needle ignored file');
    writeFileSync(join(ignoredDirectory, 'secret.txt'), 'needle ignored directory secret');
    writeFileSync(join(structuralDirectory, 'kept.txt'), 'needle structural');
    const policy = loadAiIgnorePolicy({ workspace: root });
    const globDiagnostics: AiIgnoreSuppressionDiagnostic[] = [];
    const grepDiagnostics: AiIgnoreSuppressionDiagnostic[] = [];
    const glob = globTool({
      allowedDirectories: [root],
      workingDirectory: root,
      aiIgnorePolicy: policy,
      onSuppressed: (diagnostic) => globDiagnostics.push(diagnostic),
    });
    const grep = grepTool({
      allowedDirectories: [root],
      workingDirectory: root,
      aiIgnorePolicy: policy,
      onSuppressed: (diagnostic) => grepDiagnostics.push(diagnostic),
    });

    const globResult = String(await glob.call(JSON.stringify({ pattern: '**/*.txt' })));
    const grepResult = String(await grep.call(JSON.stringify({ pattern: 'needle' })));

    expect(globResult).toContain('visible.txt');
    expect(globResult).not.toContain('ignored-file.txt');
    expect(globResult).not.toContain('secret.txt');
    expect(globResult).not.toContain('node_modules');
    expect(grepResult).toContain('needle visible');
    expect(grepResult).not.toContain('ignored file');
    expect(grepResult).not.toContain('directory secret');
    expect(globDiagnostics.map(({ surface, kind }) => ({ surface, kind }))).toEqual([
      { surface: 'glob', kind: 'directory' },
      { surface: 'glob', kind: 'file' },
    ]);
    expect(grepDiagnostics.map(({ surface, kind }) => ({ surface, kind }))).toEqual([
      { surface: 'grep', kind: 'directory' },
      { surface: 'grep', kind: 'file' },
    ]);
    expect(globDiagnostics.some((diagnostic) => diagnostic.path.includes('node_modules'))).toBe(
      false,
    );
  });

  test('suppresses a direct Grep file before reading it', async () => {
    const root = workspace();
    const ignored = join(root, 'private.txt');
    writeFileSync(join(root, '.aiignore'), 'private.txt');
    writeFileSync(ignored, 'CONTENT-MUST-NOT-BE-READ');
    const diagnostics: AiIgnoreSuppressionDiagnostic[] = [];
    const grep = grepTool({
      allowedDirectories: [root],
      workingDirectory: root,
      aiIgnorePolicy: loadAiIgnorePolicy({ workspace: root }),
      onSuppressed: (diagnostic) => diagnostics.push(diagnostic),
    });

    const result = String(await grep.call(JSON.stringify({ pattern: 'CONTENT', path: ignored })));

    expect(result).toContain('No matches');
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toMatchObject({
      code: 'aiignore-suppressed',
      surface: 'grep',
      kind: 'file',
      policyPath: join(root, '.aiignore'),
      policyLine: 1,
    });
    expect(JSON.stringify(diagnostics[0])).not.toContain('CONTENT-MUST-NOT-BE-READ');
    expect(diagnostics[0]).not.toHaveProperty('rule');
  });

  test('keeps structural and depth exclusions stronger than negated policy rules', async () => {
    const root = workspace();
    mkdirSync(join(root, 'node_modules'));
    mkdirSync(join(root, '.git'));
    mkdirSync(join(root, 'nested'));
    writeFileSync(
      join(root, '.aiignore'),
      '!node_modules/reincluded.txt\n!.git/reincluded.txt\nnested/ignored.txt',
    );
    writeFileSync(join(root, 'node_modules', 'reincluded.txt'), 'structural');
    writeFileSync(join(root, '.git', 'reincluded.txt'), 'structural');
    writeFileSync(join(root, 'nested', 'ignored.txt'), 'depth');
    const diagnostics: AiIgnoreSuppressionDiagnostic[] = [];
    const glob = globTool({
      allowedDirectories: [root],
      workingDirectory: root,
      maxDepth: 0,
      aiIgnorePolicy: loadAiIgnorePolicy({ workspace: root }),
      onSuppressed: (diagnostic) => diagnostics.push(diagnostic),
    });

    const result = String(await glob.call(JSON.stringify({ pattern: '**/*.txt' })));

    expect(result).toContain('No files matched');
    expect(diagnostics).toEqual([]);
  });

  test('excludes ignored skills from directory, explicit-file, and toolbox discovery', () => {
    const root = workspace();
    const skillsRoot = join(root, '.agents', 'skills');
    writeSkill(skillsRoot, 'visible');
    writeSkill(skillsRoot, 'hidden', 'HIDDEN-SKILL-CONTENT');
    const explicit = writeSkill(skillsRoot, 'explicit', 'EXPLICIT-SKILL-CONTENT');
    writeFileSync(
      join(root, '.aiignore'),
      '.agents/skills/hidden/\n.agents/skills/explicit/SKILL.md',
    );
    const policy = loadAiIgnorePolicy({ workspace: root });
    const diagnostics: AiIgnoreSuppressionDiagnostic[] = [];

    const loaded = loadSkillsDirectory(skillsRoot, {
      aiIgnorePolicy: policy,
      onSuppressed: (diagnostic) => diagnostics.push(diagnostic),
    });
    const explicitLoaded = collectSkills({
      files: [join(explicit, 'SKILL.md')],
      aiIgnorePolicy: policy,
      onSuppressed: (diagnostic) => diagnostics.push(diagnostic),
    });
    const toolbox = createSkillsToolbox({
      workspace: root,
      userDirectory: join(root, 'home'),
      aiIgnorePolicy: policy,
      todos: false,
    });

    expect(loaded.map((skill) => skill.name)).toEqual(['visible']);
    expect(explicitLoaded).toEqual([]);
    expect(toolbox.skills.map((skill) => skill.name)).toEqual(['visible']);
    expect(
      toolbox.skillDiagnostics.filter(({ code }) => code === 'aiignore-suppressed'),
    ).toHaveLength(2);
    expect(diagnostics.map(({ kind }) => kind)).toEqual(['file', 'directory', 'file']);
    expect(JSON.stringify(diagnostics)).not.toContain('HIDDEN-SKILL-CONTENT');
    expect(JSON.stringify(diagnostics)).not.toContain('EXPLICIT-SKILL-CONTENT');
    expect(
      SkillsToolbox.builder()
        .aiIgnorePolicy(policy)
        .onSuppressed(() => {})
        .toOptions(),
    ).toMatchObject({ aiIgnorePolicy: policy });
  });

  test('suppresses AGENTS.md before reading and selects an allowed fallback', () => {
    const root = workspace();
    const service = join(root, 'packages', 'service');
    mkdirSync(service, { recursive: true });
    writeFileSync(join(root, '.aiignore'), 'packages/AGENTS.md\npackages/service/');
    writeFileSync(join(root, 'AGENTS.md'), 'root instructions');
    const ignoredPrimary = join(root, 'packages', 'AGENTS.md');
    writeFileSync(ignoredPrimary, 'PRIMARY-CONTENT-MUST-NOT-BE-READ');
    writeFileSync(join(root, 'packages', 'FALLBACK.md'), 'fallback instructions');
    writeFileSync(join(service, 'AGENTS.md'), 'SERVICE-CONTENT-MUST-NOT-BE-READ');

    const result = discoverAgentInstructions({
      workspace: root,
      workingDirectory: service,
      fallbackFilenames: ['FALLBACK.md'],
    });

    expect(result.content).toBe('root instructions\n\nfallback instructions');
    expect(result.sources.map((source) => source.filename)).toEqual(['AGENTS.md', 'FALLBACK.md']);
    const suppressed = result.diagnostics.filter(
      (diagnostic) => diagnostic.code === 'aiignore-suppressed',
    );
    expect(suppressed).toHaveLength(2);
    expect(suppressed.every((diagnostic) => diagnostic.surface === 'agent-instructions')).toBe(
      true,
    );
    expect(JSON.stringify(suppressed)).not.toContain('PRIMARY-CONTENT-MUST-NOT-BE-READ');
    expect(JSON.stringify(suppressed)).not.toContain('SERVICE-CONTENT-MUST-NOT-BE-READ');
  });
});
