import * as fs from 'node:fs';
import { homedir } from 'node:os';
import { isAbsolute, join, resolve } from 'node:path';
import {
  type AgentInstructionSource,
  type AgentInstructionsDiagnostic,
  type DiscoverAgentInstructionsOptions,
  discoverAgentInstructions,
} from './instructions/index.ts';
import {
  type AiIgnorePolicy,
  AiIgnorePolicyError,
  type AiIgnorePolicyErrorCode,
  type AiIgnoreSuppressionDiagnostic,
  compileAiIgnorePolicy,
  loadAiIgnorePolicy,
} from './policy/index.ts';
import {
  type ResolveSkillSourcesOptions,
  resolveSkillSources,
  type SkillCatalogDiagnostic,
  type SkillCatalogDiagnosticCode,
  validateResolvedSkillCatalog,
} from './skills/index.ts';
import type { AgentSourceOrigin, ResolvedAgentSource } from './sources/index.ts';

export type AgentConfigurationAuditSeverity = 'error' | 'warning' | 'info';

export type AgentConfigurationAuditFindingCode =
  | AgentInstructionsDiagnostic['code']
  | SkillCatalogDiagnosticCode
  | 'aiignore-policy-unavailable'
  | 'migrate-vendor-instructions'
  | 'migrate-vendor-skills';

export type AgentConfigurationAuditProvenance = AgentSourceOrigin | 'policy' | 'audit';

/** A stable, content-free finding suitable for CLI or programmatic rendering. */
export interface AgentConfigurationAuditFinding {
  readonly code: AgentConfigurationAuditFindingCode;
  readonly severity: AgentConfigurationAuditSeverity;
  readonly path: string;
  readonly provenance: AgentConfigurationAuditProvenance;
  readonly message: string;
  readonly precedence?: number;
  readonly relatedPath?: string;
  readonly action?: string;
}

/** Instruction provenance without the instruction body. */
export type AuditedAgentInstructionSource = Omit<AgentInstructionSource, 'content'>;

export type VendorAgentAssetKind = 'instructions' | 'skills';

export interface VendorAgentAsset {
  readonly vendor: string;
  readonly kind: VendorAgentAssetKind;
  readonly path: string;
  readonly targetPath: string;
}

export interface AgentConfigurationMigrationOpportunity extends VendorAgentAsset {
  readonly code: 'migrate-vendor-instructions' | 'migrate-vendor-skills';
  readonly action: string;
}

export interface AgentConfigurationAuditReport {
  /** False when any audit finding has error severity. */
  readonly valid: boolean;
  readonly workspace: string;
  readonly workingDirectory: string;
  readonly instructions: {
    readonly bytes: number;
    readonly sources: readonly AuditedAgentInstructionSource[];
  };
  readonly skills: {
    readonly sources: readonly ResolvedAgentSource[];
    readonly names: readonly string[];
  };
  /** The active root policy, absent only when the workspace itself cannot be resolved. */
  readonly ignorePolicy?: AiIgnorePolicy;
  readonly suppressedSources: readonly AiIgnoreSuppressionDiagnostic[];
  readonly vendorAssets: readonly VendorAgentAsset[];
  readonly migrationOpportunities: readonly AgentConfigurationMigrationOpportunity[];
  readonly findings: readonly AgentConfigurationAuditFinding[];
}

export interface AuditAgentConfigurationOptions
  extends Pick<
      DiscoverAgentInstructionsOptions,
      'fallbackFilenames' | 'maxBytes' | 'allowedDirectories'
    >,
    Pick<ResolveSkillSourcesOptions, 'directories' | 'packages' | 'sourceMode'> {
  readonly workspace?: string;
  readonly workingDirectory?: string;
  readonly userDirectory?: string;
}

interface VendorCandidate {
  readonly vendor: string;
  readonly kind: VendorAgentAssetKind;
  readonly relativePath: string;
  readonly targetPath: string;
  readonly expectedKind: 'file' | 'directory';
}

const VENDOR_CANDIDATES: readonly VendorCandidate[] = [
  {
    vendor: 'Anthropic Claude',
    kind: 'instructions',
    relativePath: 'CLAUDE.md',
    targetPath: 'AGENTS.md',
    expectedKind: 'file',
  },
  {
    vendor: 'GitHub Copilot',
    kind: 'instructions',
    relativePath: '.github/copilot-instructions.md',
    targetPath: 'AGENTS.md',
    expectedKind: 'file',
  },
  {
    vendor: 'Cursor',
    kind: 'instructions',
    relativePath: '.cursorrules',
    targetPath: 'AGENTS.md',
    expectedKind: 'file',
  },
  {
    vendor: 'Cursor',
    kind: 'instructions',
    relativePath: '.cursor/rules',
    targetPath: 'AGENTS.md',
    expectedKind: 'directory',
  },
  {
    vendor: 'Anthropic Claude',
    kind: 'skills',
    relativePath: '.claude/skills',
    targetPath: '.agents/skills',
    expectedKind: 'directory',
  },
  {
    vendor: 'OpenAI Codex',
    kind: 'skills',
    relativePath: '.codex/skills',
    targetPath: '.agents/skills',
    expectedKind: 'directory',
  },
];

/** Audit repository agent configuration without creating agents, indexes, or files. */
export function auditAgentConfiguration(
  options: AuditAgentConfigurationOptions = {},
): AgentConfigurationAuditReport {
  const workspace = resolve(options.workspace ?? process.cwd());
  const workingDirectory = resolveFromWorkspace(
    options.workingDirectory ?? process.cwd(),
    workspace,
  );
  const userDirectory = resolve(options.userDirectory ?? homedir());
  const findings: AgentConfigurationAuditFinding[] = [];
  const ignorePolicy = resolveAuditPolicy(workspace, findings);

  const instructionResult = discoverAgentInstructions({
    workspace,
    workingDirectory,
    fallbackFilenames: options.fallbackFilenames,
    maxBytes: options.maxBytes,
    allowedDirectories: options.allowedDirectories,
    aiIgnorePolicy: ignorePolicy,
  });
  findings.push(
    ...instructionResult.diagnostics
      .filter((diagnostic) => diagnostic.code !== 'source-missing')
      .map(instructionFinding),
  );

  const skillSources = resolveSkillSources({
    workspace,
    userDirectory,
    directories: options.directories,
    packages: options.packages,
    sourceMode: options.sourceMode,
  });
  const skillSuppressions: AiIgnoreSuppressionDiagnostic[] = [];
  const skillValidation = validateResolvedSkillCatalog(skillSources, {
    aiIgnorePolicy: ignorePolicy,
    onSuppressed: (diagnostic) => skillSuppressions.push(diagnostic),
  });
  findings.push(
    ...skillValidation.diagnostics
      .filter(
        (diagnostic) =>
          diagnostic.code !== 'source-missing' ||
          (diagnostic.source.origin !== 'workspace' && diagnostic.source.origin !== 'user'),
      )
      .map(skillFinding),
  );

  const instructionSuppressions = instructionResult.diagnostics.filter(
    (diagnostic): diagnostic is AiIgnoreSuppressionDiagnostic =>
      diagnostic.code === 'aiignore-suppressed',
  );
  const suppressedSources = [...instructionSuppressions, ...skillSuppressions].sort(
    compareSuppression,
  );
  findings.push(...skillSuppressions.map(suppressionFinding));

  const vendorAssets = detectVendorAssets(workspace);
  const migrationOpportunities = vendorAssets.map(migrationOpportunity);
  findings.push(...migrationOpportunities.map(migrationFinding));
  findings.sort(compareFinding);

  return {
    valid: findings.every((finding) => finding.severity !== 'error'),
    workspace,
    workingDirectory,
    instructions: {
      bytes: instructionResult.bytes,
      sources: instructionResult.sources.map(withoutInstructionContent),
    },
    skills: {
      sources: skillSources.sources,
      names: skillValidation.skills.map((skill) => skill.name),
    },
    ignorePolicy,
    suppressedSources,
    vendorAssets,
    migrationOpportunities,
    findings,
  };
}

function resolveAuditPolicy(
  workspace: string,
  findings: AgentConfigurationAuditFinding[],
): AiIgnorePolicy | undefined {
  try {
    return loadAiIgnorePolicy({ workspace });
  } catch (error) {
    const code: AiIgnorePolicyErrorCode | undefined =
      error instanceof AiIgnorePolicyError ? error.code : undefined;
    findings.push({
      code: 'aiignore-policy-unavailable',
      severity: 'error',
      path: join(workspace, '.aiignore'),
      provenance: 'policy',
      message: error instanceof Error ? error.message : `Policy is unavailable: ${workspace}`,
      action:
        code === 'WORKSPACE_UNAVAILABLE'
          ? 'Audit an existing, readable workspace.'
          : 'Move .aiignore inside the workspace and make it readable.',
    });
    try {
      return compileAiIgnorePolicy('', { workspace });
    } catch {
      return undefined;
    }
  }
}

function instructionFinding(
  diagnostic: AgentInstructionsDiagnostic,
): AgentConfigurationAuditFinding {
  if (diagnostic.code === 'aiignore-suppressed') return suppressionFinding(diagnostic);
  return {
    code: diagnostic.code,
    severity: diagnostic.severity,
    path: diagnostic.path,
    provenance: diagnostic.origin,
    message: diagnostic.message,
    precedence: diagnostic.precedence,
    relatedPath: 'duplicateOf' in diagnostic ? diagnostic.duplicateOf : undefined,
  };
}

function skillFinding(diagnostic: SkillCatalogDiagnostic): AgentConfigurationAuditFinding {
  return {
    code: diagnostic.code,
    severity: diagnostic.severity,
    path: diagnostic.path,
    provenance: diagnostic.source.origin ?? 'audit',
    message: diagnostic.message,
    precedence: diagnostic.source.precedence,
    relatedPath: diagnostic.relatedPath,
  };
}

function suppressionFinding(
  diagnostic: AiIgnoreSuppressionDiagnostic,
): AgentConfigurationAuditFinding {
  return {
    code: diagnostic.code,
    severity: diagnostic.severity,
    path: diagnostic.path,
    provenance: 'policy',
    message: diagnostic.message,
    precedence: diagnostic.precedence,
    relatedPath: diagnostic.policyPath,
  };
}

function detectVendorAssets(workspace: string): VendorAgentAsset[] {
  const assets: VendorAgentAsset[] = [];
  for (const candidate of VENDOR_CANDIDATES) {
    const path = join(workspace, candidate.relativePath);
    let info: fs.Stats;
    try {
      info = fs.statSync(path);
    } catch {
      continue;
    }
    if (
      (candidate.expectedKind === 'file' && !info.isFile()) ||
      (candidate.expectedKind === 'directory' && !info.isDirectory())
    ) {
      continue;
    }
    assets.push({
      vendor: candidate.vendor,
      kind: candidate.kind,
      path,
      targetPath: join(workspace, candidate.targetPath),
    });
  }
  return assets.sort((left, right) => left.path.localeCompare(right.path));
}

function migrationOpportunity(asset: VendorAgentAsset): AgentConfigurationMigrationOpportunity {
  const code =
    asset.kind === 'instructions' ? 'migrate-vendor-instructions' : 'migrate-vendor-skills';
  return {
    ...asset,
    code,
    action: `Review and migrate ${asset.path} to the neutral path ${asset.targetPath}.`,
  };
}

function migrationFinding(
  opportunity: AgentConfigurationMigrationOpportunity,
): AgentConfigurationAuditFinding {
  return {
    code: opportunity.code,
    severity: 'info',
    path: opportunity.path,
    provenance: 'vendor',
    message: `${opportunity.vendor} ${opportunity.kind} are not loaded implicitly.`,
    relatedPath: opportunity.targetPath,
    action: opportunity.action,
  };
}

function withoutInstructionContent(source: AgentInstructionSource): AuditedAgentInstructionSource {
  return {
    path: source.path,
    realPath: source.realPath,
    origin: source.origin,
    precedence: source.precedence,
    kind: source.kind,
    filename: source.filename,
    directory: source.directory,
    bytes: source.bytes,
  };
}

function compareSuppression(
  left: AiIgnoreSuppressionDiagnostic,
  right: AiIgnoreSuppressionDiagnostic,
): number {
  return (
    left.path.localeCompare(right.path) ||
    left.surface.localeCompare(right.surface) ||
    left.kind.localeCompare(right.kind)
  );
}

function compareFinding(
  left: AgentConfigurationAuditFinding,
  right: AgentConfigurationAuditFinding,
): number {
  return (
    left.path.localeCompare(right.path) ||
    left.code.localeCompare(right.code) ||
    left.provenance.localeCompare(right.provenance) ||
    (left.precedence ?? Number.MAX_SAFE_INTEGER) - (right.precedence ?? Number.MAX_SAFE_INTEGER)
  );
}

function resolveFromWorkspace(path: string, workspace: string): string {
  return isAbsolute(path) ? resolve(path) : resolve(workspace, path);
}
