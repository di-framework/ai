import { afterEach, describe, expect, spyOn, test } from 'bun:test';
import * as nodeFs from 'node:fs';
import {
  chmodSync,
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import {
  type AgentConfigurationAuditReport,
  auditAgentConfiguration,
  executeAgentConfigurationMigration,
  planAgentConfigurationMigration,
  type VendorAgentAsset,
} from '../src/index.ts';

interface Fixture {
  readonly root: string;
  readonly workspace: string;
  readonly userDirectory: string;
}

function fixture(): Fixture {
  const root = mkdtempSync(join(tmpdir(), 'agent-configuration-migration-'));
  const workspace = join(root, 'workspace');
  const userDirectory = join(root, 'home');
  mkdirSync(workspace);
  mkdirSync(userDirectory);
  return { root, workspace, userDirectory };
}

function audit(value: Fixture): AgentConfigurationAuditReport {
  return auditAgentConfiguration({
    workspace: value.workspace,
    workingDirectory: value.workspace,
    userDirectory: value.userDirectory,
  });
}

function reportWithAssets(
  report: AgentConfigurationAuditReport,
  vendorAssets: readonly VendorAgentAsset[],
): AgentConfigurationAuditReport {
  return { ...report, vendorAssets };
}

function snapshot(root: string): Record<string, string> {
  const values: Record<string, string> = {};
  const stack = [root];
  while (stack.length > 0) {
    const directory = stack.pop();
    if (directory == null) break;
    for (const entry of readdirSync(directory, { withFileTypes: true }).sort((left, right) =>
      left.name.localeCompare(right.name),
    )) {
      const path = join(directory, entry.name);
      const name = relative(root, path);
      if (entry.isDirectory()) {
        values[`${name}/`] = 'directory';
        stack.push(path);
      } else if (entry.isSymbolicLink()) {
        values[name] = 'symlink';
      } else {
        values[name] = readFileSync(path, 'utf8');
      }
    }
  }
  return values;
}

describe('agent-configuration migration planning and execution', () => {
  const temps: string[] = [];
  afterEach(() => {
    for (const temp of temps.splice(0)) rmSync(temp, { recursive: true, force: true });
  });

  test('plans all requested neutral assets deterministically without mutation and dry-runs by default', () => {
    const value = fixture();
    temps.push(value.root);
    const report = audit(value);
    const before = snapshot(value.root);
    const options = {
      includeAuditOpportunities: false,
      requests: [
        { target: 'AGENTS.md', content: '# Workspace\n' },
        { target: '.agents/AGENTS.md', content: '# Agent defaults\n' },
        { target: '.agents/skills' },
        { target: '.aiignore', content: 'private/\n' },
      ],
    } as const;

    const first = planAgentConfigurationMigration(report, options);
    const second = planAgentConfigurationMigration(report, options);

    expect(first).toEqual(second);
    expect(first).toMatchObject({ version: 1, workspace: value.workspace, valid: true });
    expect(first.actions.map(({ relativeTargetPath }) => relativeTargetPath)).toEqual([
      '.agents/AGENTS.md',
      '.agents/skills',
      '.aiignore',
      'AGENTS.md',
    ]);
    expect(first.actions.every((action) => action.status === 'ready')).toBe(true);
    expect(snapshot(value.root)).toEqual(before);

    const dryRun = executeAgentConfigurationMigration(first);
    expect(dryRun).toMatchObject({ dryRun: true, success: true, changed: false });
    expect(dryRun.applied).toEqual([]);
    expect(dryRun.failed).toEqual([]);
    expect(dryRun.skipped.map(({ code }) => code)).toEqual([
      'dry-run',
      'dry-run',
      'dry-run',
      'dry-run',
    ]);
    expect(snapshot(value.root)).toEqual(before);

    const applied = executeAgentConfigurationMigration(first, { dryRun: false });
    expect(applied).toMatchObject({ dryRun: false, success: true, changed: true });
    expect(applied.applied).toHaveLength(4);
    expect(readFileSync(join(value.workspace, 'AGENTS.md'), 'utf8')).toBe('# Workspace\n');
    expect(readFileSync(join(value.workspace, '.agents', 'AGENTS.md'), 'utf8')).toBe(
      '# Agent defaults\n',
    );
    expect(readFileSync(join(value.workspace, '.aiignore'), 'utf8')).toBe('private/\n');
    expect(statSync(join(value.workspace, '.agents', 'skills')).isDirectory()).toBe(true);
    expect(existsSync(join(value.workspace, '.claude'))).toBe(false);
    expect(existsSync(join(value.workspace, '.cursor'))).toBe(false);

    const repeated = executeAgentConfigurationMigration(first, { dryRun: false });
    expect(repeated.failed).toHaveLength(4);
    expect(repeated.failed.every(({ code }) => code === 'target-changed')).toBe(true);
  });

  test('copies selected audited instructions and skill trees while retaining source modes', () => {
    const value = fixture();
    temps.push(value.root);
    const vendorInstructions = join(value.workspace, 'CLAUDE.md');
    const vendorSkills = join(value.workspace, '.claude', 'skills');
    const skill = join(vendorSkills, 'review');
    const scripts = join(skill, 'scripts');
    const empty = join(skill, 'references');
    mkdirSync(scripts, { recursive: true });
    mkdirSync(empty);
    writeFileSync(vendorInstructions, '# Vendor instructions\n');
    writeFileSync(join(skill, 'SKILL.md'), 'skill body');
    const script = join(scripts, 'review.sh');
    writeFileSync(script, '#!/bin/sh\n');
    chmodSync(script, 0o755);
    const report = audit(value);
    const before = snapshot(value.root);

    const instructionsOnly = planAgentConfigurationMigration(report, {
      opportunityPaths: [vendorInstructions],
    });
    expect(instructionsOnly.actions).toHaveLength(1);
    expect(instructionsOnly.actions[0]).toMatchObject({
      relativeTargetPath: 'AGENTS.md',
      operation: 'write-file',
      source: { kind: 'file', path: vendorInstructions },
    });

    const plan = planAgentConfigurationMigration(report);
    expect(plan.valid).toBe(true);
    expect(plan.actions.map(({ relativeTargetPath }) => relativeTargetPath)).toEqual([
      '.agents/skills',
      '.agents/skills/review',
      '.agents/skills/review/references',
      '.agents/skills/review/scripts',
      '.agents/skills/review/scripts/review.sh',
      '.agents/skills/review/SKILL.md',
      'AGENTS.md',
    ]);
    expect(snapshot(value.root)).toEqual(before);

    const result = executeAgentConfigurationMigration(plan, { dryRun: false });
    expect(result.failed).toEqual([]);
    expect(result.applied).toHaveLength(7);
    expect(readFileSync(join(value.workspace, 'AGENTS.md'), 'utf8')).toBe(
      '# Vendor instructions\n',
    );
    expect(
      readFileSync(join(value.workspace, '.agents', 'skills', 'review', 'SKILL.md'), 'utf8'),
    ).toBe('skill body');
    expect(
      statSync(join(value.workspace, '.agents', 'skills', 'review', 'scripts', 'review.sh')).mode &
        0o777,
    ).toBe(0o755);
    expect(snapshot(join(value.workspace, '.claude', 'skills'))).toEqual(snapshot(vendorSkills));
  });

  test('detects collisions and only replaces a real file through an explicit recoverable action', () => {
    const value = fixture();
    temps.push(value.root);
    const target = join(value.workspace, 'AGENTS.md');
    writeFileSync(target, 'existing');
    const report = audit(value);

    const collision = planAgentConfigurationMigration(report, {
      includeAuditOpportunities: false,
      requests: [{ target: 'AGENTS.md', content: 'replacement' }],
    });
    expect(collision.valid).toBe(false);
    expect(collision.actions[0]).toMatchObject({ status: 'failed', code: 'target-exists' });
    const rejected = executeAgentConfigurationMigration(collision, { dryRun: false });
    expect(rejected.failed).toHaveLength(1);
    expect(readFileSync(target, 'utf8')).toBe('existing');

    const replace = planAgentConfigurationMigration(report, {
      includeAuditOpportunities: false,
      requests: [{ target: 'AGENTS.md', content: 'replacement', replaceExisting: true }],
    });
    expect(replace.actions[0]).toMatchObject({ status: 'ready', operation: 'replace-file' });
    const result = executeAgentConfigurationMigration(replace, { dryRun: false });
    const backup = `${target}.di-framework-backup`;
    expect(result.applied[0]).toMatchObject({ status: 'applied', backupPath: backup });
    expect(readFileSync(target, 'utf8')).toBe('replacement');
    expect(readFileSync(backup, 'utf8')).toBe('existing');

    const current = planAgentConfigurationMigration(audit(value), {
      includeAuditOpportunities: false,
      requests: [{ target: 'AGENTS.md', content: 'replacement' }],
    });
    expect(current.actions[0]).toMatchObject({ status: 'skipped', code: 'already-current' });
    expect(executeAgentConfigurationMigration(current, { dryRun: false }).skipped).toHaveLength(1);
    writeFileSync(target, 'changed after skipped plan');
    expect(executeAgentConfigurationMigration(current).failed[0]).toMatchObject({
      code: 'target-changed',
    });
  });

  test('reports duplicate targets, unsafe symlinks, incompatible kinds, and unsupported sources', () => {
    const value = fixture();
    temps.push(value.root);
    const outside = join(value.root, 'outside');
    mkdirSync(outside);
    symlinkSync(outside, join(value.workspace, '.agents'));
    mkdirSync(join(value.workspace, 'AGENTS.md'));
    const base = audit(value);
    const duplicates = planAgentConfigurationMigration(base, {
      includeAuditOpportunities: false,
      requests: [
        { target: 'AGENTS.md', content: 'one' },
        { target: 'AGENTS.md', content: 'two' },
        { target: '.agents/AGENTS.md', content: 'unsafe' },
        { target: '.aiignore', content: 'safe' },
      ],
    });
    expect(duplicates.actions.filter(({ code }) => code === 'plan-target-collision')).toHaveLength(
      2,
    );
    expect(duplicates.actions.find(({ asset }) => asset === '.agents/AGENTS.md')).toMatchObject({
      code: 'target-symlink-unsafe',
    });
    expect(duplicates.actions.find(({ asset }) => asset === 'AGENTS.md')).toMatchObject({
      code: 'plan-target-collision',
    });
    expect(
      duplicates.actions.some(
        ({ code, targetPath }) =>
          code === 'target-kind-conflict' && targetPath === join(value.workspace, 'AGENTS.md'),
      ),
    ).toBe(false);

    const incompatible = planAgentConfigurationMigration(base, {
      includeAuditOpportunities: false,
      requests: [{ target: 'AGENTS.md', content: 'one' }],
    });
    expect(incompatible.actions[0]).toMatchObject({ code: 'target-kind-conflict' });

    rmSync(join(value.workspace, '.agents'));
    mkdirSync(join(value.workspace, '.agents'));
    writeFileSync(join(value.workspace, '.agents', 'skills'), 'not a directory');
    const skillsConflict = planAgentConfigurationMigration(audit(value), {
      includeAuditOpportunities: false,
      requests: [{ target: '.agents/skills' }],
    });
    expect(skillsConflict.actions[0]).toMatchObject({
      operation: 'create-directory',
      code: 'target-kind-conflict',
    });
    rmSync(join(value.workspace, '.agents', 'skills'));
    mkdirSync(join(value.workspace, '.agents', 'skills'));
    const skillsCurrent = planAgentConfigurationMigration(audit(value), {
      includeAuditOpportunities: false,
      requests: [{ target: '.agents/skills' }],
    });
    expect(skillsCurrent.actions[0]).toMatchObject({ status: 'skipped', code: 'already-exists' });

    const cursorRules = join(value.workspace, '.cursor', 'rules');
    mkdirSync(cursorRules, { recursive: true });
    const unsupported = planAgentConfigurationMigration(audit(value), {
      opportunityPaths: [cursorRules],
    });
    expect(unsupported.actions[0]).toMatchObject({
      status: 'failed',
      code: 'source-kind-unsupported',
    });

    const firstInstructions = join(value.workspace, 'first-vendor.md');
    const secondInstructions = join(value.workspace, 'second-vendor.md');
    writeFileSync(firstInstructions, 'first');
    writeFileSync(secondInstructions, 'second');
    const vendorCollision = planAgentConfigurationMigration(
      reportWithAssets(base, [
        {
          vendor: 'first',
          kind: 'instructions',
          path: firstInstructions,
          targetPath: join(value.workspace, 'AGENTS.md'),
        },
        {
          vendor: 'second',
          kind: 'instructions',
          path: secondInstructions,
          targetPath: join(value.workspace, 'AGENTS.md'),
        },
      ]),
    );
    expect(vendorCollision.actions.every(({ code }) => code === 'plan-target-collision')).toBe(
      true,
    );

    const outsideTarget = reportWithAssets(base, [
      {
        vendor: 'test',
        kind: 'instructions',
        path: join(value.workspace, '.aiignore'),
        targetPath: join(value.root, 'not-neutral'),
      },
    ]);
    expect(planAgentConfigurationMigration(outsideTarget).actions[0]).toMatchObject({
      code: 'target-outside-neutral-paths',
    });
  });

  test('fails safely when sources or targets change after planning and reports partial outcomes', () => {
    const value = fixture();
    temps.push(value.root);
    const source = join(value.workspace, 'CLAUDE.md');
    writeFileSync(source, 'first');
    const sourcePlan = planAgentConfigurationMigration(audit(value));
    writeFileSync(source, 'changed');
    const changedSource = executeAgentConfigurationMigration(sourcePlan, { dryRun: false });
    expect(changedSource.failed[0]).toMatchObject({ code: 'source-changed' });
    expect(existsSync(join(value.workspace, 'AGENTS.md'))).toBe(false);

    const report = audit(value);
    const partial = planAgentConfigurationMigration(report, {
      includeAuditOpportunities: false,
      requests: [{ target: '.aiignore', content: 'ignore' }, { target: '.agents/skills' }],
    });
    writeFileSync(join(value.workspace, '.aiignore'), 'late collision');
    const partialResult = executeAgentConfigurationMigration(partial, { dryRun: false });
    expect(partialResult).toMatchObject({ success: false, changed: true });
    expect(partialResult.applied).toHaveLength(1);
    expect(partialResult.failed[0]).toMatchObject({ code: 'target-changed' });
  });

  test('refuses backup collisions and symbolic-link sources without changing either side', () => {
    const value = fixture();
    temps.push(value.root);
    const target = join(value.workspace, 'AGENTS.md');
    writeFileSync(target, 'old');
    const replace = planAgentConfigurationMigration(audit(value), {
      includeAuditOpportunities: false,
      requests: [{ target: 'AGENTS.md', content: 'new', replaceExisting: true }],
    });
    writeFileSync(`${target}.di-framework-backup`, 'reserved');
    expect(executeAgentConfigurationMigration(replace, { dryRun: false }).failed[0]).toMatchObject({
      code: 'backup-exists',
    });
    expect(readFileSync(target, 'utf8')).toBe('old');
    expect(
      planAgentConfigurationMigration(audit(value), {
        includeAuditOpportunities: false,
        requests: [{ target: 'AGENTS.md', content: 'new', replaceExisting: true }],
      }).actions[0],
    ).toMatchObject({ operation: 'replace-file', code: 'backup-exists', status: 'failed' });

    const sourceTarget = join(value.workspace, 'linked-instructions');
    symlinkSync(target, sourceTarget);
    const symlinkReport = reportWithAssets(audit(value), [
      {
        vendor: 'test',
        kind: 'instructions',
        path: sourceTarget,
        targetPath: join(value.workspace, '.agents', 'AGENTS.md'),
      },
    ]);
    expect(planAgentConfigurationMigration(symlinkReport).actions[0]).toMatchObject({
      code: 'source-symlink-unsupported',
    });
    expect(lstatSync(sourceTarget).isSymbolicLink()).toBe(true);
  });

  test('turns missing, unreadable, wrong-kind, and nested symbolic-link sources into typed plan failures', () => {
    const value = fixture();
    temps.push(value.root);
    const base = audit(value);
    const target = join(value.workspace, 'AGENTS.md');
    const missing = join(value.workspace, 'missing-vendor.md');
    const missingPlan = planAgentConfigurationMigration(
      reportWithAssets(base, [
        { vendor: 'test', kind: 'instructions', path: missing, targetPath: target },
      ]),
    );
    expect(missingPlan.actions[0]).toMatchObject({ code: 'source-unavailable' });

    const wrongSkills = join(value.workspace, 'vendor-skills-file');
    writeFileSync(wrongSkills, 'file');
    expect(
      planAgentConfigurationMigration(
        reportWithAssets(base, [
          {
            vendor: 'test',
            kind: 'skills',
            path: wrongSkills,
            targetPath: join(value.workspace, '.agents', 'skills'),
          },
        ]),
      ).actions[0],
    ).toMatchObject({ code: 'source-kind-unsupported' });

    const vendorSkills = join(value.workspace, 'vendor-skills');
    mkdirSync(vendorSkills);
    writeFileSync(join(vendorSkills, 'real.md'), 'real');
    const unreadableNested = join(vendorSkills, 'unreadable.md');
    writeFileSync(unreadableNested, 'secret');
    chmodSync(unreadableNested, 0o000);
    symlinkSync(join(vendorSkills, 'real.md'), join(vendorSkills, 'linked.md'));
    const nestedSymlink = planAgentConfigurationMigration(
      reportWithAssets(base, [
        {
          vendor: 'test',
          kind: 'skills',
          path: vendorSkills,
          targetPath: join(value.workspace, '.agents', 'skills'),
        },
      ]),
    );
    expect(nestedSymlink.actions.some(({ code }) => code === 'source-symlink-unsupported')).toBe(
      true,
    );
    expect(nestedSymlink.actions.some(({ code }) => code === 'source-unavailable')).toBe(true);
    chmodSync(unreadableNested, 0o600);

    const special = join(vendorSkills, 'special');
    const fifo = Bun.spawnSync(['mkfifo', special]);
    expect(fifo.exitCode).toBe(0);
    const specialPlan = planAgentConfigurationMigration(
      reportWithAssets(base, [
        {
          vendor: 'test',
          kind: 'skills',
          path: vendorSkills,
          targetPath: join(value.workspace, '.agents', 'skills'),
        },
      ]),
    );
    expect(specialPlan.actions.some(({ code }) => code === 'source-kind-unsupported')).toBe(true);

    const unreadable = join(value.workspace, 'unreadable.md');
    writeFileSync(unreadable, 'secret');
    chmodSync(unreadable, 0o000);
    const unreadablePlan = planAgentConfigurationMigration(
      reportWithAssets(base, [
        { vendor: 'test', kind: 'instructions', path: unreadable, targetPath: target },
      ]),
    );
    chmodSync(unreadable, 0o600);
    expect(unreadablePlan.actions[0]).toMatchObject({ code: 'source-unavailable' });

    const unreadableDirectory = join(value.workspace, 'unreadable-skills');
    mkdirSync(unreadableDirectory);
    chmodSync(unreadableDirectory, 0o000);
    const unreadableDirectoryPlan = planAgentConfigurationMigration(
      reportWithAssets(base, [
        {
          vendor: 'test',
          kind: 'skills',
          path: unreadableDirectory,
          targetPath: join(value.workspace, '.agents', 'skills'),
        },
      ]),
    );
    chmodSync(unreadableDirectory, 0o700);
    expect(unreadableDirectoryPlan.actions.some(({ code }) => code === 'source-unavailable')).toBe(
      true,
    );

    const unreadableTarget = join(value.workspace, '.aiignore');
    writeFileSync(unreadableTarget, 'current');
    chmodSync(unreadableTarget, 0o000);
    const unreadableTargetPlan = planAgentConfigurationMigration(base, {
      includeAuditOpportunities: false,
      requests: [{ target: '.aiignore', content: 'replacement' }],
    });
    chmodSync(unreadableTarget, 0o600);
    expect(unreadableTargetPlan.actions[0]).toMatchObject({ code: 'target-kind-conflict' });
  });

  test('rejects malformed or no-longer-applicable plans and reports filesystem apply failures', () => {
    const value = fixture();
    temps.push(value.root);
    const report = audit(value);
    const base = planAgentConfigurationMigration(report, {
      includeAuditOpportunities: false,
      requests: [{ target: '.aiignore', content: 'private/' }],
    });
    const action = base.actions[0];
    expect(action).toBeDefined();
    if (action == null) throw new Error('expected action');

    const missingSource = {
      ...base,
      actions: [{ ...action, source: undefined }],
    };
    expect(
      executeAgentConfigurationMigration(missingSource, { dryRun: false }).failed[0],
    ).toMatchObject({ code: 'apply-failed' });

    const outsidePath = join(value.root, 'outside-target');
    const outside = {
      ...base,
      actions: [
        {
          ...action,
          targetPath: outsidePath,
          relativeTargetPath: '../outside-target',
          targetBefore: { kind: 'missing' as const },
        },
      ],
    };
    expect(executeAgentConfigurationMigration(outside, { dryRun: false }).failed[0]).toMatchObject({
      code: 'target-outside-neutral-paths',
    });

    const stage = join(value.workspace, `..aiignore.${action.id}.${process.pid}.tmp`);
    writeFileSync(stage, 'occupied');
    expect(executeAgentConfigurationMigration(base, { dryRun: false }).failed[0]).toMatchObject({
      code: 'apply-failed',
    });
    expect(existsSync(join(value.workspace, '.aiignore'))).toBe(false);

    rmSync(stage);
    const directoryPlan = planAgentConfigurationMigration(report, {
      includeAuditOpportunities: false,
      requests: [{ target: '.agents/skills' }],
    });
    writeFileSync(join(value.workspace, '.agents'), 'parent collision');
    expect(
      executeAgentConfigurationMigration(directoryPlan, { dryRun: false }).failed[0],
    ).toMatchObject({ code: 'apply-failed' });
  });

  test('cleans atomic staging and restores recovery backups when the final link fails', () => {
    const value = fixture();
    temps.push(value.root);
    const target = join(value.workspace, 'AGENTS.md');
    writeFileSync(target, 'old');
    const plan = planAgentConfigurationMigration(audit(value), {
      includeAuditOpportunities: false,
      requests: [{ target: 'AGENTS.md', content: 'new', replaceExisting: true }],
    });
    const link = spyOn(nodeFs, 'linkSync').mockImplementation(() => {
      throw new Error('simulated final-link failure');
    });
    const result = executeAgentConfigurationMigration(plan, { dryRun: false });
    link.mockRestore();

    expect(result.failed[0]).toMatchObject({ code: 'apply-failed' });
    expect(readFileSync(target, 'utf8')).toBe('old');
    expect(existsSync(`${target}.di-framework-backup`)).toBe(false);
    expect(readdirSync(value.workspace).some((name) => name.endsWith(`.${process.pid}.tmp`))).toBe(
      false,
    );

    const source = join(value.workspace, 'CLAUDE.md');
    writeFileSync(source, 'source');
    const sourcePlan = planAgentConfigurationMigration(audit(value), {
      opportunityPaths: [source],
      replaceExisting: true,
    });
    chmodSync(source, 0o000);
    expect(
      executeAgentConfigurationMigration(sourcePlan, { dryRun: false }).failed[0],
    ).toMatchObject({ code: 'source-changed' });
    chmodSync(source, 0o600);
  });
});
