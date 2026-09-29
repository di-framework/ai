import { afterEach, describe, expect, test } from 'bun:test';
import {
  chmodSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import { auditAgentConfiguration } from '../src/index.ts';

interface RepositoryFixture {
  readonly root: string;
  readonly workspace: string;
  readonly userDirectory: string;
}

function repository(): RepositoryFixture {
  const root = mkdtempSync(join(tmpdir(), 'agent-configuration-audit-'));
  const workspace = join(root, 'workspace');
  const userDirectory = join(root, 'home');
  mkdirSync(workspace);
  mkdirSync(userDirectory);
  return { root, workspace, userDirectory };
}

function writeSkill(skillsRoot: string, directoryName: string, name = directoryName): string {
  const directory = join(skillsRoot, directoryName);
  mkdirSync(directory, { recursive: true });
  writeFileSync(
    join(directory, 'SKILL.md'),
    `---\nname: ${name}\ndescription: Use ${name} for repository audit tests.\n---\n${name} body`,
  );
  return directory;
}

function snapshot(root: string): Record<string, string> {
  const files: Record<string, string> = {};
  const stack = [root];
  while (stack.length > 0) {
    const directory = stack.pop();
    if (directory == null) break;
    for (const entry of readdirSync(directory, { withFileTypes: true }).sort((left, right) =>
      left.name.localeCompare(right.name),
    )) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) stack.push(path);
      else if (entry.isFile()) files[relative(root, path)] = readFileSync(path, 'utf8');
    }
  }
  return files;
}

describe('repository agent-configuration audit', () => {
  const temps: string[] = [];
  afterEach(() => {
    for (const temp of temps.splice(0)) rmSync(temp, { recursive: true, force: true });
  });

  test('returns a stable typed report for a clean neutral repository without mutation', () => {
    const fixture = repository();
    temps.push(fixture.root);
    const workingDirectory = join(fixture.workspace, 'packages', 'service');
    mkdirSync(workingDirectory, { recursive: true });
    writeFileSync(join(fixture.workspace, 'AGENTS.md'), 'workspace instructions');
    writeFileSync(join(fixture.workspace, '.aiignore'), 'private/\n');
    writeSkill(join(fixture.workspace, '.agents', 'skills'), 'workspace-review');
    writeSkill(join(fixture.userDirectory, '.agents', 'skills'), 'user-review');
    const before = snapshot(fixture.root);

    const options = {
      workspace: fixture.workspace,
      workingDirectory,
      userDirectory: fixture.userDirectory,
    } as const;
    const first = auditAgentConfiguration(options);
    const second = auditAgentConfiguration(options);

    expect(first).toEqual(second);
    expect(first.valid).toBe(true);
    expect(first.instructions.sources).toHaveLength(1);
    expect(first.instructions.sources[0]).not.toHaveProperty('content');
    expect(first.skills.sources.map((source) => source.origin)).toEqual(['workspace', 'user']);
    expect(first.skills.names).toEqual(['workspace-review', 'user-review']);
    expect(first.ignorePolicy).toMatchObject({ source: { exists: true }, rules: [{ line: 1 }] });
    expect(first.suppressedSources).toEqual([]);
    expect(first.vendorAssets).toEqual([]);
    expect(snapshot(fixture.root)).toEqual(before);
  });

  test('reports mixed validation failures and policy-suppressed sources without content', () => {
    const fixture = repository();
    temps.push(fixture.root);
    const skillsRoot = join(fixture.workspace, '.agents', 'skills');
    writeSkill(skillsRoot, 'visible');
    writeSkill(skillsRoot, 'hidden');
    const malformed = join(skillsRoot, 'malformed');
    mkdirSync(malformed);
    writeFileSync(join(malformed, 'SKILL.md'), '---\nname:\n  - invalid\n---\nMALFORMED-BODY');
    mkdirSync(join(fixture.userDirectory, '.agents', 'skills'), { recursive: true });
    writeFileSync(join(fixture.workspace, 'AGENTS.md'), 'visible instructions');
    writeFileSync(join(fixture.workspace, 'PRIVATE.md'), 'PRIVATE-INSTRUCTION-CONTENT');
    writeFileSync(join(fixture.workspace, '.aiignore'), '.agents/skills/hidden/\nPRIVATE.md\n');

    const report = auditAgentConfiguration({
      workspace: fixture.workspace,
      workingDirectory: fixture.workspace,
      userDirectory: fixture.userDirectory,
      fallbackFilenames: ['PRIVATE.md'],
    });
    const serialized = JSON.stringify(report);

    expect(report.valid).toBe(false);
    expect(report.skills.names).toContain('visible');
    expect(report.skills.names).not.toContain('hidden');
    expect(report.findings.some((finding) => finding.code === 'skill-frontmatter-invalid')).toBe(
      true,
    );
    expect(report.suppressedSources.map((diagnostic) => diagnostic.surface)).toEqual([
      'skill-discovery',
      'agent-instructions',
    ]);
    expect(serialized).not.toContain('hidden body');
    expect(serialized).not.toContain('PRIVATE-INSTRUCTION-CONTENT');
    expect(serialized).not.toContain('MALFORMED-BODY');
  });

  test('reports conflicting precedence and vendor migrations without loading vendor assets', () => {
    const fixture = repository();
    temps.push(fixture.root);
    const first = join(fixture.workspace, 'first');
    const second = join(fixture.workspace, 'second');
    const missing = join(fixture.workspace, 'missing-explicit');
    writeSkill(first, 'one', 'shared');
    writeSkill(first, 'two', 'shared');
    writeSkill(second, 'shared');
    writeFileSync(join(fixture.workspace, 'AGENTS.md'), 'neutral');
    const vendorSkill = join(fixture.workspace, '.claude', 'skills', 'vendor-only');
    mkdirSync(vendorSkill, { recursive: true });
    writeFileSync(join(vendorSkill, 'SKILL.md'), 'VENDOR-CONTENT-MUST-NOT-BE-LOADED');
    writeFileSync(join(fixture.workspace, 'CLAUDE.md'), 'VENDOR-INSTRUCTIONS-MUST-NOT-BE-LOADED');

    const report = auditAgentConfiguration({
      workspace: fixture.workspace,
      workingDirectory: fixture.workspace,
      userDirectory: fixture.userDirectory,
      directories: [first, second, missing],
      sourceMode: 'replace',
    });
    const codes = report.findings.map((finding) => finding.code);
    const serialized = JSON.stringify(report);

    expect(report.skills.names).toEqual(['shared']);
    expect(codes).toContain('skill-duplicate');
    expect(codes).toContain('skill-shadowed');
    expect(codes).toContain('source-missing');
    expect(report.findings.find((finding) => finding.code === 'source-missing')).toMatchObject({
      path: missing,
      provenance: 'explicit',
      severity: 'error',
    });
    expect(report.vendorAssets.map(({ kind }) => kind)).toEqual(['skills', 'instructions']);
    expect(report.migrationOpportunities.map(({ code }) => code)).toEqual([
      'migrate-vendor-skills',
      'migrate-vendor-instructions',
    ]);
    expect(
      report.findings
        .filter((finding) => finding.code.startsWith('migrate-vendor-'))
        .every(
          (finding) =>
            finding.provenance === 'vendor' &&
            finding.severity === 'info' &&
            finding.action != null,
        ),
    ).toBe(true);
    expect(serialized).not.toContain('VENDOR-CONTENT-MUST-NOT-BE-LOADED');
    expect(serialized).not.toContain('VENDOR-INSTRUCTIONS-MUST-NOT-BE-LOADED');
    expect(
      report.findings.some(
        (finding) =>
          finding.code === 'skill-frontmatter-invalid' && finding.path.includes('.claude'),
      ),
    ).toBe(false);
  });

  test('fails closed for restricted instruction, skill, and ignore-policy paths', () => {
    const fixture = repository();
    temps.push(fixture.root);
    const allowed = join(fixture.workspace, 'allowed');
    const outside = join(fixture.root, 'outside-skills');
    mkdirSync(allowed);
    writeSkill(outside, 'outside');
    writeFileSync(join(fixture.workspace, 'AGENTS.md'), 'outside allowed scope');
    mkdirSync(join(fixture.workspace, '.agents'));
    symlinkSync(outside, join(fixture.workspace, '.agents', 'skills'));
    mkdirSync(join(fixture.userDirectory, '.agents', 'skills'), { recursive: true });
    const policy = join(fixture.workspace, '.aiignore');
    writeFileSync(policy, 'secret/');
    chmodSync(policy, 0o000);

    const report = auditAgentConfiguration({
      workspace: fixture.workspace,
      workingDirectory: allowed,
      userDirectory: fixture.userDirectory,
      allowedDirectories: [allowed],
    });
    chmodSync(policy, 0o600);

    expect(report.valid).toBe(false);
    expect(report.ignorePolicy).toBeDefined();
    expect(report.findings.map((finding) => finding.code)).toContain('aiignore-policy-unavailable');
    expect(report.findings.map((finding) => finding.code)).toContain(
      'instructions-outside-allowed-directories',
    );
    expect(report.findings.map((finding) => finding.code)).toContain('source-outside-boundary');
    expect(
      report.findings.every(
        (finding) =>
          typeof finding.path === 'string' &&
          typeof finding.provenance === 'string' &&
          typeof finding.severity === 'string' &&
          typeof finding.code === 'string',
      ),
    ).toBe(true);

    const missingWorkspace = auditAgentConfiguration({
      workspace: join(fixture.root, 'missing-workspace'),
      workingDirectory: join(fixture.root, 'missing-workspace'),
      userDirectory: fixture.userDirectory,
    });
    expect(missingWorkspace.ignorePolicy).toBeUndefined();
    expect(missingWorkspace.findings.map((finding) => finding.code)).toContain(
      'aiignore-policy-unavailable',
    );
  });
});
