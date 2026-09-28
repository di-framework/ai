import { describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  agentSkill,
  collectSkills,
  createSkillsToolbox,
  DEFAULT_SKILL_DIRECTORY_CANDIDATES,
  resolveSkillPackageDirectories,
  resolveSkillSources,
  SkillsToolbox,
} from '../src/index.ts';

function roots(): { workspace: string; userDirectory: string; explicit: string } {
  const root = mkdtempSync(join(tmpdir(), 'ai-utils-skill-sources-'));
  const workspace = join(root, 'workspace');
  const userDirectory = join(root, 'home');
  const explicit = join(root, 'explicit');
  mkdirSync(workspace);
  mkdirSync(userDirectory);
  mkdirSync(explicit);
  return { workspace, userDirectory, explicit };
}

function writeSkill(root: string, name: string, content: string): string {
  const directory = join(root, name);
  mkdirSync(directory, { recursive: true });
  writeFileSync(
    join(directory, 'SKILL.md'),
    `---\nname: ${name}\ndescription: Use ${name} when the user asks for ${name}.\n---\n${content}`,
  );
  return directory;
}

describe('neutral skill source resolution', () => {
  test('merge orders explicit, workspace, and user roots and reports shadowed skills', () => {
    const { workspace, userDirectory, explicit } = roots();
    const workspaceSkills = join(workspace, '.agents', 'skills');
    const userSkills = join(userDirectory, '.agents', 'skills');
    writeSkill(explicit, 'shared', 'explicit wins');
    writeSkill(workspaceSkills, 'shared', 'workspace loses');
    writeSkill(workspaceSkills, 'workspace-only', 'workspace');
    writeSkill(userSkills, 'shared', 'user loses');
    writeSkill(userSkills, 'user-only', 'user');

    const toolbox = createSkillsToolbox({
      directories: [explicit],
      workspace,
      userDirectory,
      sourceMode: 'merge',
      todos: false,
    });

    expect(toolbox.skillSources.map(({ origin, precedence }) => ({ origin, precedence }))).toEqual([
      { origin: 'explicit', precedence: 0 },
      { origin: 'workspace', precedence: 1 },
      { origin: 'user', precedence: 2 },
    ]);
    expect(toolbox.skills.map((skill) => skill.name)).toEqual([
      'shared',
      'workspace-only',
      'user-only',
    ]);
    expect(toolbox.skills[0]?.content).toContain('explicit wins');
    expect(toolbox.skillDiagnostics).toEqual([
      {
        code: 'skill-duplicate',
        severity: 'warning',
        name: 'shared',
        keptSource: join(explicit, 'shared'),
        ignoredSource: join(workspaceSkills, 'shared'),
      },
      {
        code: 'skill-duplicate',
        severity: 'warning',
        name: 'shared',
        keptSource: join(explicit, 'shared'),
        ignoredSource: join(userSkills, 'shared'),
      },
    ]);
  });

  test('replace uses only explicit roots and empty arrays no longer disable defaults', () => {
    const { workspace, userDirectory, explicit } = roots();
    writeSkill(explicit, 'explicit-only', 'explicit');
    writeSkill(join(workspace, '.agents', 'skills'), 'workspace-only', 'workspace');
    writeSkill(join(userDirectory, '.agents', 'skills'), 'user-only', 'user');

    const replaced = resolveSkillSources({
      directories: [explicit],
      workspace,
      userDirectory,
      sourceMode: 'replace',
    });
    expect(replaced.sources.map((source) => source.origin)).toEqual(['explicit']);

    const mergedEmpty = resolveSkillSources({
      directories: [],
      workspace,
      userDirectory,
    });
    expect(mergedEmpty.sources.map((source) => source.origin)).toEqual(['workspace', 'user']);
    expect(SkillsToolbox.builder().userDirectory(userDirectory).toOptions().userDirectory).toBe(
      userDirectory,
    );
  });

  test('only neutral automatic roots are candidates', () => {
    const { workspace, userDirectory } = roots();
    writeSkill(join(workspace, '.claude', 'skills'), 'legacy', 'must not load');
    writeSkill(join(userDirectory, '.codex', 'skills'), 'vendor', 'must not load');

    expect(DEFAULT_SKILL_DIRECTORY_CANDIDATES).toEqual(['.agents/skills', '~/.agents/skills']);
    const resolution = resolveSkillSources({ workspace, userDirectory });
    expect(resolution.sources).toEqual([]);
    expect(resolution.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
      'source-missing',
      'source-missing',
    ]);
  });

  test('package discovery prefers declarations, then .agents/skills, then skills', () => {
    const { workspace } = roots();
    const declaredPackage = join(workspace, 'declared-package');
    const neutralPackage = join(workspace, 'neutral-package');
    const conventionalPackage = join(workspace, 'conventional-package');
    const declared = join(declaredPackage, 'catalog');
    mkdirSync(declared, { recursive: true });
    mkdirSync(join(declaredPackage, '.agents', 'skills'), { recursive: true });
    mkdirSync(join(neutralPackage, '.agents', 'skills'), { recursive: true });
    mkdirSync(join(neutralPackage, 'skills'), { recursive: true });
    mkdirSync(join(conventionalPackage, 'skills'), { recursive: true });
    writeFileSync(
      join(declaredPackage, 'package.json'),
      JSON.stringify({ name: 'declared', skills: './catalog' }),
    );
    writeFileSync(join(neutralPackage, 'package.json'), JSON.stringify({ name: 'neutral' }));
    writeFileSync(
      join(conventionalPackage, 'package.json'),
      JSON.stringify({ name: 'conventional' }),
    );

    expect(
      resolveSkillPackageDirectories(
        [declaredPackage, neutralPackage, conventionalPackage],
        workspace,
      ),
    ).toEqual([
      declared,
      join(neutralPackage, '.agents', 'skills'),
      join(conventionalPackage, 'skills'),
    ]);
    expect(
      resolveSkillSources({
        packages: [declaredPackage],
        workspace,
        sourceMode: 'replace',
      }).sources[0]?.origin,
    ).toBe('package');
  });

  test('collectSkills keeps the first duplicate and reports both sources', () => {
    const first = agentSkill({
      name: 'same',
      description: 'first',
      content: 'first',
      basePath: '/first',
    });
    const second = agentSkill({
      name: 'same',
      description: 'second',
      content: 'second',
      basePath: '/second',
    });
    const diagnostics: unknown[] = [];

    expect(
      collectSkills({
        skills: [first, second],
        onDuplicate: (diagnostic) => diagnostics.push(diagnostic),
      }),
    ).toEqual([first]);
    expect(diagnostics).toEqual([
      {
        code: 'skill-duplicate',
        severity: 'warning',
        name: 'same',
        keptSource: '/first',
        ignoredSource: '/second',
      },
    ]);
  });
});
