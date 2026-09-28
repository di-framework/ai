import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import { basename, dirname, join, relative, resolve, sep } from 'node:path';
import type {
  AgentConfigurationAuditReport,
  VendorAgentAsset,
} from './audit-agent-configuration.ts';

export type NeutralAgentAssetPath =
  | 'AGENTS.md'
  | '.agents/AGENTS.md'
  | '.agents/skills'
  | '.aiignore';

export type AgentConfigurationMigrationPlanCode =
  | 'ready'
  | 'already-current'
  | 'already-exists'
  | 'source-unavailable'
  | 'source-kind-unsupported'
  | 'source-symlink-unsupported'
  | 'target-exists'
  | 'target-kind-conflict'
  | 'target-symlink-unsafe'
  | 'target-outside-neutral-paths'
  | 'backup-exists'
  | 'plan-target-collision';

export type AgentConfigurationMigrationActionStatus = 'ready' | 'skipped' | 'failed';
export type AgentConfigurationMigrationOperation =
  | 'create-directory'
  | 'write-file'
  | 'replace-file';

export type AgentConfigurationMigrationRequest =
  | {
      readonly target: 'AGENTS.md' | '.agents/AGENTS.md' | '.aiignore';
      readonly content: string;
      readonly replaceExisting?: boolean;
    }
  | {
      readonly target: '.agents/skills';
    };

export interface PlanAgentConfigurationMigrationOptions {
  /** Include audit-discovered vendor assets. Defaults to true. */
  readonly includeAuditOpportunities?: boolean;
  /** Limit audit migration to these exact source paths. */
  readonly opportunityPaths?: readonly string[];
  /** Explicit neutral files or the neutral skills directory to initialize. */
  readonly requests?: readonly AgentConfigurationMigrationRequest[];
  /** Make vendor-file collisions explicit replace actions instead of failures. */
  readonly replaceExisting?: boolean;
}

export interface AgentConfigurationMigrationInlineSource {
  readonly kind: 'inline';
  readonly content: string;
  readonly bytes: number;
  readonly sha256: string;
  readonly mode: number;
}

export interface AgentConfigurationMigrationFileSource {
  readonly kind: 'file';
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
  readonly mode: number;
}

export type AgentConfigurationMigrationSource =
  | AgentConfigurationMigrationInlineSource
  | AgentConfigurationMigrationFileSource;

export interface AgentConfigurationMigrationTargetState {
  readonly kind: 'missing' | 'file' | 'directory' | 'symlink' | 'other';
  readonly bytes?: number;
  readonly sha256?: string;
}

export interface AgentConfigurationMigrationAction {
  readonly id: string;
  readonly asset: NeutralAgentAssetPath;
  readonly operation: AgentConfigurationMigrationOperation;
  readonly targetPath: string;
  readonly relativeTargetPath: string;
  readonly source?: AgentConfigurationMigrationSource;
  readonly backupPath?: string;
  readonly targetBefore: AgentConfigurationMigrationTargetState;
  readonly status: AgentConfigurationMigrationActionStatus;
  readonly code: AgentConfigurationMigrationPlanCode;
  readonly message: string;
}

export interface AgentConfigurationMigrationPlan {
  readonly version: 1;
  readonly workspace: string;
  readonly valid: boolean;
  readonly actions: readonly AgentConfigurationMigrationAction[];
}

export type AgentConfigurationMigrationExecutionCode =
  | AgentConfigurationMigrationPlanCode
  | 'dry-run'
  | 'source-changed'
  | 'target-changed'
  | 'apply-failed';

export interface AgentConfigurationMigrationActionResult {
  readonly actionId: string;
  readonly operation: AgentConfigurationMigrationOperation;
  readonly targetPath: string;
  readonly status: 'applied' | 'skipped' | 'failed';
  readonly code: AgentConfigurationMigrationExecutionCode;
  readonly message: string;
  readonly backupPath?: string;
}

export interface ExecuteAgentConfigurationMigrationOptions {
  /** Execution is a dry run unless callers explicitly pass false. */
  readonly dryRun?: boolean;
}

export interface AgentConfigurationMigrationExecutionResult {
  readonly dryRun: boolean;
  readonly success: boolean;
  readonly changed: boolean;
  readonly applied: readonly AgentConfigurationMigrationActionResult[];
  readonly skipped: readonly AgentConfigurationMigrationActionResult[];
  readonly failed: readonly AgentConfigurationMigrationActionResult[];
}

interface DraftAction extends Omit<AgentConfigurationMigrationAction, 'id'> {}

/** Build a stable migration plan without changing the workspace. */
export function planAgentConfigurationMigration(
  report: AgentConfigurationAuditReport,
  options: PlanAgentConfigurationMigrationOptions = {},
): AgentConfigurationMigrationPlan {
  const workspace = resolve(report.workspace);
  const drafts: DraftAction[] = [];
  if (options.includeAuditOpportunities !== false) {
    const selected = options.opportunityPaths?.map((path) => resolve(workspace, path));
    const selectedPaths = selected == null ? undefined : new Set(selected);
    for (const asset of [...report.vendorAssets].sort(compareVendorAsset)) {
      if (selectedPaths != null && !selectedPaths.has(resolve(asset.path))) continue;
      drafts.push(...planVendorAsset(workspace, asset, options.replaceExisting === true));
    }
  }
  for (const request of options.requests ?? []) drafts.push(planRequest(workspace, request));

  const actions = finalizeActions(workspace, drafts);
  return {
    version: 1,
    workspace,
    valid: actions.every((action) => action.status !== 'failed'),
    actions,
  };
}

/** Apply an exact plan. Dry-run is the safe default and performs no writes. */
export function executeAgentConfigurationMigration(
  plan: AgentConfigurationMigrationPlan,
  options: ExecuteAgentConfigurationMigrationOptions = {},
): AgentConfigurationMigrationExecutionResult {
  const dryRun = options.dryRun !== false;
  const applied: AgentConfigurationMigrationActionResult[] = [];
  const skipped: AgentConfigurationMigrationActionResult[] = [];
  const failed: AgentConfigurationMigrationActionResult[] = [];

  for (const action of plan.actions) {
    if (action.status === 'failed') {
      failed.push(actionResult(action, 'failed', action.code, action.message));
      continue;
    }
    const unsafe = unsafeTargetCode(resolve(plan.workspace), action.targetPath);
    if (unsafe != null) {
      failed.push(actionResult(action, 'failed', unsafe.code, unsafe.message));
      continue;
    }
    if (!sameTargetState(targetState(action.targetPath), action.targetBefore)) {
      failed.push(
        actionResult(
          action,
          'failed',
          'target-changed',
          `Target changed after planning: ${action.targetPath}.`,
        ),
      );
      continue;
    }
    if (action.status === 'skipped') {
      skipped.push(actionResult(action, 'skipped', action.code, action.message));
      continue;
    }
    if (dryRun) {
      skipped.push(
        actionResult(action, 'skipped', 'dry-run', `Would ${describeOperation(action)}.`),
      );
      continue;
    }
    const outcome = applyAction(resolve(plan.workspace), action);
    if (outcome.status === 'applied') applied.push(outcome);
    else failed.push(outcome);
  }

  return {
    dryRun,
    success: failed.length === 0,
    changed: applied.length > 0,
    applied,
    skipped,
    failed,
  };
}

function planRequest(workspace: string, request: AgentConfigurationMigrationRequest): DraftAction {
  const targetPath = join(workspace, request.target);
  if (request.target === '.agents/skills') {
    return planDirectory(workspace, '.agents/skills', targetPath);
  }
  return planFile(
    workspace,
    request.target,
    targetPath,
    inlineSource(request.content),
    request.replaceExisting === true,
  );
}

function planVendorAsset(
  workspace: string,
  asset: VendorAgentAsset,
  replaceExisting: boolean,
): DraftAction[] {
  const sourcePath = resolve(asset.path);
  const targetPath = resolve(asset.targetPath);
  const neutralAsset = neutralAssetForTarget(workspace, targetPath);
  if (neutralAsset == null) {
    return [
      failedDraft(
        '.agents/skills',
        targetPath,
        'target-outside-neutral-paths',
        `Refusing non-neutral target ${targetPath}.`,
      ),
    ];
  }
  const sourceInfo = lstat(sourcePath);
  if (sourceInfo == null) {
    return [
      failedDraft(
        neutralAsset,
        targetPath,
        'source-unavailable',
        `Migration source is unavailable: ${sourcePath}.`,
      ),
    ];
  }
  if (sourceInfo.isSymbolicLink()) {
    return [
      failedDraft(
        neutralAsset,
        targetPath,
        'source-symlink-unsupported',
        `Migration sources cannot be symbolic links: ${sourcePath}.`,
      ),
    ];
  }
  if (asset.kind === 'instructions') {
    if (!sourceInfo.isFile()) {
      return [
        failedDraft(
          neutralAsset,
          targetPath,
          'source-kind-unsupported',
          `Instruction migration requires a regular file: ${sourcePath}.`,
        ),
      ];
    }
    const source = fileSource(sourcePath, sourceInfo);
    return source == null
      ? [
          failedDraft(
            neutralAsset,
            targetPath,
            'source-unavailable',
            `Migration source is unreadable: ${sourcePath}.`,
          ),
        ]
      : [planFile(workspace, neutralAsset, targetPath, source, replaceExisting)];
  }
  if (!sourceInfo.isDirectory()) {
    return [
      failedDraft(
        neutralAsset,
        targetPath,
        'source-kind-unsupported',
        `Skill migration requires a directory: ${sourcePath}.`,
      ),
    ];
  }
  return planDirectoryTree(workspace, sourcePath, targetPath, replaceExisting);
}

function planDirectoryTree(
  workspace: string,
  sourceRoot: string,
  targetRoot: string,
  replaceExisting: boolean,
): DraftAction[] {
  const actions: DraftAction[] = [planDirectory(workspace, '.agents/skills', targetRoot)];
  const visit = (sourceDirectory: string): void => {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(sourceDirectory, { withFileTypes: true });
    } catch {
      actions.push(
        failedDraft(
          '.agents/skills',
          join(targetRoot, relative(sourceRoot, sourceDirectory)),
          'source-unavailable',
          `Migration source directory is unreadable: ${sourceDirectory}.`,
        ),
      );
      return;
    }
    entries.sort((left, right) => left.name.localeCompare(right.name));
    for (const entry of entries) {
      const sourcePath = join(sourceDirectory, entry.name);
      const relativePath = relative(sourceRoot, sourcePath);
      const targetPath = join(targetRoot, relativePath);
      if (entry.isSymbolicLink()) {
        actions.push(
          failedDraft(
            '.agents/skills',
            targetPath,
            'source-symlink-unsupported',
            `Migration sources cannot contain symbolic links: ${sourcePath}.`,
          ),
        );
      } else if (entry.isDirectory()) {
        actions.push(planDirectory(workspace, '.agents/skills', targetPath));
        visit(sourcePath);
      } else if (entry.isFile()) {
        const info = lstat(sourcePath);
        const source = info == null ? undefined : fileSource(sourcePath, info);
        actions.push(
          source == null
            ? failedDraft(
                '.agents/skills',
                targetPath,
                'source-unavailable',
                `Migration source is unavailable: ${sourcePath}.`,
              )
            : planFile(workspace, '.agents/skills', targetPath, source, replaceExisting),
        );
      } else {
        actions.push(
          failedDraft(
            '.agents/skills',
            targetPath,
            'source-kind-unsupported',
            `Migration source is not a regular file or directory: ${sourcePath}.`,
          ),
        );
      }
    }
  };
  visit(sourceRoot);
  return actions;
}

function planDirectory(
  workspace: string,
  asset: NeutralAgentAssetPath,
  targetPath: string,
): DraftAction {
  const unsafe = unsafeTargetCode(workspace, targetPath);
  if (unsafe != null) return failedDraft(asset, targetPath, unsafe.code, unsafe.message);
  const targetBefore = targetState(targetPath);
  if (targetBefore.kind === 'missing') {
    return readyDraft(
      asset,
      'create-directory',
      targetPath,
      targetBefore,
      undefined,
      `Create neutral directory ${targetPath}.`,
    );
  }
  if (targetBefore.kind === 'directory') {
    return skippedDraft(
      asset,
      'create-directory',
      targetPath,
      targetBefore,
      'already-exists',
      `Neutral directory already exists: ${targetPath}.`,
    );
  }
  return failedDraft(
    asset,
    targetPath,
    targetBefore.kind === 'symlink' ? 'target-symlink-unsafe' : 'target-kind-conflict',
    `Neutral directory target has an incompatible existing entry: ${targetPath}.`,
    targetBefore,
    'create-directory',
  );
}

function planFile(
  workspace: string,
  asset: NeutralAgentAssetPath,
  targetPath: string,
  source: AgentConfigurationMigrationSource,
  replaceExisting: boolean,
): DraftAction {
  const unsafe = unsafeTargetCode(workspace, targetPath);
  if (unsafe != null) {
    return failedDraft(asset, targetPath, unsafe.code, unsafe.message, undefined, 'write-file');
  }
  const targetBefore = targetState(targetPath);
  if (targetBefore.kind === 'missing') {
    return readyDraft(
      asset,
      'write-file',
      targetPath,
      targetBefore,
      source,
      `Write neutral file ${targetPath}.`,
    );
  }
  if (targetBefore.kind === 'file' && targetBefore.sha256 === source.sha256) {
    return skippedDraft(
      asset,
      'write-file',
      targetPath,
      targetBefore,
      'already-current',
      `Neutral file already has the planned content: ${targetPath}.`,
      source,
    );
  }
  if (targetBefore.kind === 'file' && replaceExisting) {
    const backupPath = `${targetPath}.di-framework-backup`;
    if (targetState(backupPath).kind !== 'missing') {
      return failedDraft(
        asset,
        targetPath,
        'backup-exists',
        `Recovery backup already exists: ${backupPath}.`,
        targetBefore,
        'replace-file',
        source,
      );
    }
    return readyDraft(
      asset,
      'replace-file',
      targetPath,
      targetBefore,
      source,
      `Replace ${targetPath} and retain a recovery backup.`,
    );
  }
  const code =
    targetBefore.kind === 'file'
      ? 'target-exists'
      : targetBefore.kind === 'symlink'
        ? 'target-symlink-unsafe'
        : 'target-kind-conflict';
  return failedDraft(
    asset,
    targetPath,
    code,
    `Neutral file target collides with an existing entry: ${targetPath}.`,
    targetBefore,
    'write-file',
    source,
  );
}

function finalizeActions(
  workspace: string,
  drafts: readonly DraftAction[],
): AgentConfigurationMigrationAction[] {
  const uniqueDirectories = new Map<string, DraftAction>();
  const others: DraftAction[] = [];
  for (const draft of drafts) {
    if (draft.operation === 'create-directory' && draft.status !== 'failed') {
      const existing = uniqueDirectories.get(draft.targetPath);
      if (existing == null || (existing.status === 'skipped' && draft.status === 'ready')) {
        uniqueDirectories.set(draft.targetPath, draft);
      }
    } else {
      others.push(draft);
    }
  }
  const combined = [...uniqueDirectories.values(), ...others].sort(compareDraft);
  const targetCounts = new Map<string, number>();
  for (const action of combined) {
    if (action.operation !== 'create-directory') {
      targetCounts.set(action.targetPath, (targetCounts.get(action.targetPath) ?? 0) + 1);
    }
  }
  return combined.map((draft, index) => {
    const collision = (targetCounts.get(draft.targetPath) ?? 0) > 1;
    const finalized = collision
      ? {
          ...draft,
          status: 'failed' as const,
          code: 'plan-target-collision' as const,
          message: `Multiple planned actions target ${draft.targetPath}; select one source explicitly.`,
        }
      : draft;
    return {
      id: `migration-${String(index + 1).padStart(3, '0')}`,
      ...finalized,
      relativeTargetPath: relative(workspace, finalized.targetPath).split(sep).join('/'),
      backupPath:
        finalized.operation === 'replace-file'
          ? `${finalized.targetPath}.di-framework-backup`
          : undefined,
    };
  });
}

function applyAction(
  workspace: string,
  action: AgentConfigurationMigrationAction,
): AgentConfigurationMigrationActionResult {
  if (action.operation === 'create-directory') {
    try {
      createDirectorySafely(workspace, action.targetPath);
      return actionResult(action, 'applied', 'ready', `Created ${action.targetPath}.`);
    } catch {
      return actionResult(
        action,
        'failed',
        'apply-failed',
        `Could not create ${action.targetPath}.`,
      );
    }
  }

  const source = action.source;
  if (source == null) {
    return actionResult(action, 'failed', 'apply-failed', 'Planned file content is missing.');
  }
  const content = readPlannedSource(source);
  if (content == null) {
    return actionResult(
      action,
      'failed',
      'source-changed',
      `Source changed after planning${source.kind === 'file' ? `: ${source.path}` : ''}.`,
    );
  }
  const backupPath = action.backupPath ?? `${action.targetPath}.di-framework-backup`;
  if (action.operation === 'replace-file' && targetState(backupPath).kind !== 'missing') {
    return actionResult(
      action,
      'failed',
      'backup-exists',
      `Recovery backup already exists: ${backupPath}.`,
    );
  }
  try {
    createDirectorySafely(workspace, dirname(action.targetPath));
    writeFileAtomically(action, content, backupPath);
    return actionResult(
      action,
      'applied',
      'ready',
      `${action.operation === 'replace-file' ? 'Replaced' : 'Wrote'} ${action.targetPath}.`,
      action.operation === 'replace-file' ? backupPath : undefined,
    );
  } catch {
    return actionResult(
      action,
      'failed',
      'apply-failed',
      `Could not ${describeOperation(action)}.`,
    );
  }
}

function createDirectorySafely(workspace: string, targetPath: string): void {
  const relativePath = relative(workspace, targetPath);
  let current = workspace;
  for (const part of relativePath.split(sep).filter(Boolean)) {
    current = join(current, part);
    const info = lstat(current);
    if (info == null) fs.mkdirSync(current);
    else if (!info.isDirectory() || info.isSymbolicLink()) throw new Error('unsafe directory');
  }
}

function writeFileAtomically(
  action: AgentConfigurationMigrationAction,
  content: Buffer,
  backupPath: string,
): void {
  const targetPath = action.targetPath;
  const stagePath = join(
    dirname(targetPath),
    `.${basename(targetPath)}.${action.id}.${process.pid}.tmp`,
  );
  let staged = false;
  let backedUp = false;
  try {
    fs.writeFileSync(stagePath, content, {
      flag: 'wx',
      mode: action.source?.mode ?? 0o644,
    });
    staged = true;
    if (action.operation === 'replace-file') {
      fs.renameSync(targetPath, backupPath);
      backedUp = true;
    }
    fs.linkSync(stagePath, targetPath);
    staged = false;
    try {
      fs.unlinkSync(stagePath);
    } catch {}
  } catch (error) {
    if (staged) {
      try {
        fs.unlinkSync(stagePath);
      } catch {}
    }
    if (backedUp && targetState(targetPath).kind === 'missing') {
      try {
        fs.renameSync(backupPath, targetPath);
      } catch {}
    }
    throw error;
  }
}

function readPlannedSource(source: AgentConfigurationMigrationSource): Buffer | undefined {
  if (source.kind === 'inline') {
    const content = Buffer.from(source.content);
    return fingerprint(content).sha256 === source.sha256 ? content : undefined;
  }
  const info = lstat(source.path);
  if (info == null || !info.isFile() || info.isSymbolicLink()) return undefined;
  try {
    const content = fs.readFileSync(source.path);
    const value = fingerprint(content);
    return value.bytes === source.bytes && value.sha256 === source.sha256 ? content : undefined;
  } catch {
    return undefined;
  }
}

function unsafeTargetCode(
  workspace: string,
  targetPath: string,
): { code: 'target-outside-neutral-paths' | 'target-symlink-unsafe'; message: string } | undefined {
  if (neutralAssetForTarget(workspace, targetPath) == null) {
    return {
      code: 'target-outside-neutral-paths',
      message: `Target is outside the neutral agent paths: ${targetPath}.`,
    };
  }
  let current = dirname(targetPath);
  while (current !== workspace) {
    const info = lstat(current);
    if (info?.isSymbolicLink()) {
      return {
        code: 'target-symlink-unsafe',
        message: `Target ancestor is a symbolic link: ${current}.`,
      };
    }
    const parent = dirname(current);
    if (parent === current || relative(workspace, current).startsWith(`..${sep}`)) break;
    current = parent;
  }
  return undefined;
}

function neutralAssetForTarget(
  workspace: string,
  targetPath: string,
): NeutralAgentAssetPath | undefined {
  const relativePath = relative(workspace, resolve(targetPath));
  if (relativePath === 'AGENTS.md') return 'AGENTS.md';
  if (relativePath === join('.agents', 'AGENTS.md')) return '.agents/AGENTS.md';
  if (relativePath === '.aiignore') return '.aiignore';
  if (
    relativePath === join('.agents', 'skills') ||
    relativePath.startsWith(`${join('.agents', 'skills')}${sep}`)
  ) {
    return '.agents/skills';
  }
  return undefined;
}

function inlineSource(content: string): AgentConfigurationMigrationInlineSource {
  return { kind: 'inline', content, ...fingerprint(Buffer.from(content)), mode: 0o644 };
}

function fileSource(
  path: string,
  info: fs.Stats,
): AgentConfigurationMigrationFileSource | undefined {
  try {
    const content = fs.readFileSync(path);
    return { kind: 'file', path, ...fingerprint(content), mode: info.mode & 0o777 };
  } catch {
    return undefined;
  }
}

function fingerprint(content: Buffer): { bytes: number; sha256: string } {
  return {
    bytes: content.byteLength,
    sha256: createHash('sha256').update(content).digest('hex'),
  };
}

function targetState(path: string): AgentConfigurationMigrationTargetState {
  const info = lstat(path);
  if (info == null) return { kind: 'missing' };
  if (info.isSymbolicLink()) return { kind: 'symlink' };
  if (info.isDirectory()) return { kind: 'directory' };
  if (!info.isFile()) return { kind: 'other' };
  try {
    const content = fs.readFileSync(path);
    return { kind: 'file', ...fingerprint(content) };
  } catch {
    return { kind: 'other' };
  }
}

function lstat(path: string): fs.Stats | undefined {
  try {
    return fs.lstatSync(path);
  } catch {
    return undefined;
  }
}

function sameTargetState(
  left: AgentConfigurationMigrationTargetState,
  right: AgentConfigurationMigrationTargetState,
): boolean {
  return left.kind === right.kind && left.bytes === right.bytes && left.sha256 === right.sha256;
}

function readyDraft(
  asset: NeutralAgentAssetPath,
  operation: AgentConfigurationMigrationOperation,
  targetPath: string,
  targetBefore: AgentConfigurationMigrationTargetState,
  source: AgentConfigurationMigrationSource | undefined,
  message: string,
): DraftAction {
  return {
    asset,
    operation,
    targetPath,
    relativeTargetPath: targetPath,
    source,
    targetBefore,
    status: 'ready',
    code: 'ready',
    message,
  };
}

function skippedDraft(
  asset: NeutralAgentAssetPath,
  operation: AgentConfigurationMigrationOperation,
  targetPath: string,
  targetBefore: AgentConfigurationMigrationTargetState,
  code: 'already-current' | 'already-exists',
  message: string,
  source?: AgentConfigurationMigrationSource,
): DraftAction {
  return {
    asset,
    operation,
    targetPath,
    relativeTargetPath: targetPath,
    source,
    targetBefore,
    status: 'skipped',
    code,
    message,
  };
}

function failedDraft(
  asset: NeutralAgentAssetPath,
  targetPath: string,
  code: Exclude<
    AgentConfigurationMigrationPlanCode,
    'ready' | 'already-current' | 'already-exists'
  >,
  message: string,
  targetBefore: AgentConfigurationMigrationTargetState = targetState(targetPath),
  operation: AgentConfigurationMigrationOperation = 'write-file',
  source?: AgentConfigurationMigrationSource,
): DraftAction {
  return {
    asset,
    operation,
    targetPath,
    relativeTargetPath: targetPath,
    source,
    targetBefore,
    status: 'failed',
    code,
    message,
  };
}

function actionResult(
  action: AgentConfigurationMigrationAction,
  status: 'applied' | 'skipped' | 'failed',
  code: AgentConfigurationMigrationExecutionCode,
  message: string,
  backupPath?: string,
): AgentConfigurationMigrationActionResult {
  return {
    actionId: action.id,
    operation: action.operation,
    targetPath: action.targetPath,
    status,
    code,
    message,
    backupPath,
  };
}

function describeOperation(action: AgentConfigurationMigrationAction): string {
  if (action.operation === 'create-directory') return `create ${action.targetPath}`;
  if (action.operation === 'replace-file') return `replace ${action.targetPath}`;
  return `write ${action.targetPath}`;
}

function compareVendorAsset(left: VendorAgentAsset, right: VendorAgentAsset): number {
  return left.path.localeCompare(right.path) || left.targetPath.localeCompare(right.targetPath);
}

function compareDraft(left: DraftAction, right: DraftAction): number {
  const leftSourcePath = left.source?.kind === 'file' ? left.source.path : '';
  const rightSourcePath = right.source?.kind === 'file' ? right.source.path : '';
  return (
    left.targetPath.localeCompare(right.targetPath) ||
    left.operation.localeCompare(right.operation) ||
    leftSourcePath.localeCompare(rightSourcePath)
  );
}
