import { describe, expect, test } from 'bun:test';
import { chmodSync, mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  agentSkill,
  resolveSkillSources,
  validateResolvedSkillCatalog,
  validateSkillCatalog,
  validateSkillDefinition,
  validateSkillDirectory,
  validateSkillsDirectory,
} from '../src/index.ts';

function root(): string {
  return mkdtempSync(join(tmpdir(), 'skill-catalog-validation-'));
}

function writeSkill(
  skillsRoot: string,
  directoryName: string,
  name = directoryName,
  body = 'Use this skill.',
): string {
  const directory = join(skillsRoot, directoryName);
  mkdirSync(directory, { recursive: true });
  writeFileSync(
    join(directory, 'SKILL.md'),
    `---\nname: ${name}\ndescription: Use ${name} for validation tests.\n---\n${body}`,
  );
  return directory;
}

function codes(result: ReturnType<typeof validateSkillDirectory>): string[] {
  return result.diagnostics.map((diagnostic) => diagnostic.code);
}

describe('skill catalog validation', () => {
  test('validates an in-memory definition without creating an agent or index', () => {
    const result = validateSkillDefinition(
      agentSkill({
        name: 'Invalid--Name',
        description: '',
        content: 'body',
        basePath: '/skills/different-name',
      }),
    );

    expect(result.valid).toBe(false);
    expect(codes(result)).toEqual([
      'skill-name-invalid',
      'skill-description-invalid',
      'skill-name-directory-mismatch',
    ]);
    expect(
      result.diagnostics.every((diagnostic) => diagnostic.source.path === '/skills/different-name'),
    ).toBe(true);
  });

  test('reports malformed frontmatter and a missing entrypoint', () => {
    const workspace = root();
    const malformed = join(workspace, 'malformed');
    const empty = join(workspace, 'empty');
    mkdirSync(malformed);
    mkdirSync(empty);
    writeFileSync(join(malformed, 'SKILL.md'), '---\nname: malformed\n  bad: indent\nbody');

    expect(codes(validateSkillDirectory(malformed))).toContain('skill-frontmatter-invalid');
    expect(codes(validateSkillDirectory(empty))).toEqual(['skill-entrypoint-missing']);
  });

  test('reports invalid directory shapes and broken entrypoints', () => {
    const workspace = root();
    const file = join(workspace, 'file');
    const brokenDirectory = join(workspace, 'broken-directory');
    const brokenEntrypoint = join(workspace, 'broken-entrypoint');
    writeFileSync(file, 'not a directory');
    symlinkSync(join(workspace, 'missing-directory'), brokenDirectory);
    mkdirSync(brokenEntrypoint);
    symlinkSync(join(workspace, 'missing-skill'), join(brokenEntrypoint, 'SKILL.md'));

    expect(codes(validateSkillDirectory(join(workspace, 'missing')))).toEqual([
      'skill-entrypoint-missing',
    ]);
    expect(codes(validateSkillDirectory(file))).toEqual(['skill-entrypoint-missing']);
    expect(codes(validateSkillDirectory(brokenDirectory))).toEqual([
      'skill-resource-broken-symlink',
    ]);
    expect(codes(validateSkillDirectory(brokenEntrypoint))).toEqual([
      'skill-resource-broken-symlink',
    ]);
  });

  test('validates frontmatter field types and YAML shapes', () => {
    const workspace = root();
    const typed = join(workspace, 'typed');
    const sequence = join(workspace, 'sequence');
    const invalidYaml = join(workspace, 'invalid-yaml');
    mkdirSync(typed);
    mkdirSync(sequence);
    mkdirSync(invalidYaml);
    writeFileSync(
      join(typed, 'SKILL.md'),
      [
        '---',
        'name:',
        '  - typed',
        'description:',
        '  - invalid',
        'compatibility: ""',
        'metadata: invalid',
        '---',
        'body',
      ].join('\n'),
    );
    writeFileSync(join(sequence, 'SKILL.md'), '---\n- invalid\n---\nbody');
    writeFileSync(join(invalidYaml, 'SKILL.md'), '---\nname: invalid-yaml\n  bad: indent\n---');

    expect(
      codes(validateSkillDirectory(typed)).filter((code) => code === 'skill-frontmatter-invalid'),
    ).toHaveLength(4);
    expect(codes(validateSkillDirectory(sequence))).toContain('skill-frontmatter-invalid');
    expect(codes(validateSkillDirectory(invalidYaml))).toContain('skill-frontmatter-invalid');
  });

  test('validates missing, unreadable, broken, and escaping resources', () => {
    const workspace = root();
    const skill = writeSkill(
      workspace,
      'resources',
      'resources',
      [
        '[missing](references/missing.md)',
        '[outside](../outside.txt)',
        '`scripts/broken.sh`',
        '`assets/unreadable.txt`',
      ].join('\n'),
    );
    mkdirSync(join(skill, 'references'));
    mkdirSync(join(skill, 'scripts'));
    mkdirSync(join(skill, 'assets'));
    writeFileSync(join(workspace, 'outside.txt'), 'outside');
    symlinkSync(join(skill, 'does-not-exist'), join(skill, 'scripts', 'broken.sh'));
    const unreadable = join(skill, 'assets', 'unreadable.txt');
    writeFileSync(unreadable, 'secret');
    chmodSync(unreadable, 0o000);

    const result = validateSkillDirectory(skill);
    chmodSync(unreadable, 0o600);

    expect(codes(result)).toContain('skill-resource-missing');
    expect(codes(result)).toContain('skill-resource-unreadable');
    expect(codes(result)).toContain('skill-resource-broken-symlink');
    expect(codes(result)).toContain('skill-resource-outside-directory');
  });

  test('checks resource directory access, real symlink boundaries, and safe reference parsing', () => {
    const workspace = root();
    const skill = writeSkill(
      workspace,
      'resource-edges',
      'resource-edges',
      [
        '[anchor](#section)',
        '[web](https://example.com/file)',
        '[absolute](/tmp/file)',
        '[encoded](references/%ZZ)',
        '`assets/external.txt`',
      ].join('\n'),
    );
    const references = join(skill, 'references');
    const assets = join(skill, 'assets');
    mkdirSync(references);
    mkdirSync(assets);
    const outside = join(workspace, 'outside-resource.txt');
    writeFileSync(outside, 'outside');
    symlinkSync(outside, join(assets, 'external.txt'));
    chmodSync(references, 0o000);

    const result = validateSkillDirectory(skill);
    chmodSync(references, 0o700);

    expect(codes(result)).toContain('skill-resource-unreadable');
    expect(codes(result)).toContain('skill-resource-outside-directory');
  });

  test('distinguishes same-source duplicates from lower-precedence shadows', () => {
    const workspace = root();
    const explicit = join(workspace, 'explicit');
    const lower = join(workspace, 'lower');
    writeSkill(explicit, 'first-directory', 'shared');
    writeSkill(explicit, 'second-directory', 'shared');
    writeSkill(lower, 'shared');
    const resolution = resolveSkillSources({
      workspace,
      directories: [explicit, lower],
      sourceMode: 'replace',
    });

    const result = validateResolvedSkillCatalog(resolution);

    expect(result.skills.map((skill) => skill.name)).toEqual(['shared']);
    expect(codes(result)).toContain('skill-duplicate');
    expect(codes(result)).toContain('skill-shadowed');
    const shadow = result.diagnostics.find((diagnostic) => diagnostic.code === 'skill-shadowed');
    expect(shadow?.source.precedence).toBe(1);
    expect(shadow?.relatedPath).toMatch(/(?:first|second)-directory/);
    expect(validateSkillsDirectory(lower).skills.map((skill) => skill.name)).toEqual(['shared']);
  });

  test('reports a source that becomes unreadable after resolution', () => {
    const workspace = root();
    const missing = join(workspace, 'removed-after-resolution');
    const result = validateResolvedSkillCatalog({
      directories: [missing],
      diagnostics: [],
      sources: [
        {
          path: missing,
          realPath: missing,
          origin: 'explicit',
          precedence: 0,
          kind: 'directory',
        },
      ],
    });

    expect(codes(result)).toEqual(['source-unreadable']);
  });

  test('uses runtime resolution and preserves unreadable and broken source diagnostics', () => {
    const workspace = root();
    const unreadable = join(workspace, 'unreadable');
    const broken = join(workspace, 'broken');
    mkdirSync(unreadable);
    chmodSync(unreadable, 0o000);
    symlinkSync(join(workspace, 'missing-target'), broken);

    const result = validateSkillCatalog({
      workspace,
      directories: [unreadable, broken],
      sourceMode: 'replace',
    });
    chmodSync(unreadable, 0o700);

    expect(codes(result)).toEqual(['source-unreadable', 'source-broken-symlink']);
    expect(result.diagnostics.map((diagnostic) => diagnostic.source.precedence)).toEqual([0, 1]);
  });
});
