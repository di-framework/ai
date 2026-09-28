import * as fs from 'node:fs';
import { constants } from 'node:fs';
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import {
  type AiIgnorePolicy,
  type AiIgnoreSuppressionDiagnostic,
  aiIgnoreSuppressionDiagnostic,
  evaluateAiIgnorePath,
} from '../policy/index.ts';
import { nodeErrnoCode } from '../sandbox/fs-error.ts';
import type {
  AgentSourceDiagnosticCode,
  AgentSourceOrigin,
  ResolvedAgentSource,
} from '../sources/resolve-agent-sources.ts';
import { parseYaml, type YamlMap } from '../yaml/parse-yaml.ts';
import type { AgentSkill } from './parse-skill-markdown.ts';
import { parseSkillMarkdown } from './parse-skill-markdown.ts';
import {
  type ResolvedSkillSources,
  type ResolveSkillSourcesOptions,
  resolveSkillSources,
} from './resolve-skill-sources.ts';
import { validateSkillDescription, validateSkillName } from './validate-skill.ts';

const SKIP_DIR_NAMES = new Set(['node_modules', '.git', 'dist', 'coverage']);
const RESOURCE_DIRECTORY_NAMES = new Set(['scripts', 'references', 'assets']);

export type SkillCatalogDiagnosticCode =
  | AgentSourceDiagnosticCode
  | 'skill-frontmatter-invalid'
  | 'skill-name-invalid'
  | 'skill-description-invalid'
  | 'skill-name-directory-mismatch'
  | 'skill-entrypoint-missing'
  | 'skill-duplicate'
  | 'skill-shadowed'
  | 'skill-resource-missing'
  | 'skill-resource-unreadable'
  | 'skill-resource-broken-symlink'
  | 'skill-resource-outside-directory';

export interface SkillDiagnosticSource {
  readonly path: string;
  readonly realPath?: string;
  readonly origin?: AgentSourceOrigin;
  readonly precedence?: number;
}

/** A machine-readable validation finding. Messages contain no terminal formatting. */
export interface SkillCatalogDiagnostic {
  readonly code: SkillCatalogDiagnosticCode;
  readonly severity: 'error' | 'warning';
  readonly message: string;
  readonly path: string;
  readonly skillName?: string;
  readonly source: SkillDiagnosticSource;
  readonly relatedPath?: string;
}

export interface SkillValidationResult {
  readonly valid: boolean;
  readonly skills: readonly AgentSkill[];
  readonly diagnostics: readonly SkillCatalogDiagnostic[];
}

export interface ValidateSkillDefinitionOptions {
  /** Filesystem or logical source represented by the skill. */
  readonly path?: string;
  /** Catalog source metadata, when the definition came from resolved discovery. */
  readonly source?: SkillDiagnosticSource;
  /** Defaults to true for filesystem-backed skills. */
  readonly matchDirectoryName?: boolean;
}

export interface ValidateResolvedSkillCatalogOptions {
  /** Optional discovery policy applied before traversing or reading SKILL.md files. */
  readonly aiIgnorePolicy?: AiIgnorePolicy;
  /** Receives content-free records for policy-suppressed skill sources. */
  readonly onSuppressed?: (diagnostic: AiIgnoreSuppressionDiagnostic) => void;
}

/** Validate one parsed skill without creating an agent or semantic index. */
export function validateSkillDefinition(
  skill: AgentSkill,
  options: ValidateSkillDefinitionOptions = {},
): SkillValidationResult {
  const path = resolveDiagnosticPath(options.path ?? skill.basePath ?? '.');
  const source = options.source ?? { path };
  const diagnostics: SkillCatalogDiagnostic[] = [];
  const add = (code: SkillCatalogDiagnosticCode, message: string, relatedPath?: string): void => {
    diagnostics.push({
      code,
      severity: 'error',
      message,
      path,
      skillName: skill.name || undefined,
      source,
      relatedPath,
    });
  };

  const nameError = validateSkillName(skill.name);
  if (nameError) add('skill-name-invalid', nameError);
  const descriptionError = validateSkillDescription(skill.description);
  if (descriptionError) add('skill-description-invalid', descriptionError);

  const matchDirectoryName = options.matchDirectoryName ?? skill.basePath !== '.';
  if (matchDirectoryName && skill.basePath && skill.basePath !== '.') {
    const directoryName = basename(skill.basePath);
    if (skill.name && directoryName !== skill.name) {
      add(
        'skill-name-directory-mismatch',
        `name "${skill.name}" must match the skill directory name "${directoryName}"`,
        skill.basePath,
      );
    }
  }

  return validationResult([skill], diagnostics);
}

/** Validate a single skill directory, including its entrypoint and bundled resources. */
export function validateSkillDirectory(directory: string): SkillValidationResult {
  const directoryPath = resolve(directory);
  const source: SkillDiagnosticSource = { path: directoryPath };
  const entrypoint = join(directoryPath, 'SKILL.md');
  const diagnostics: SkillCatalogDiagnostic[] = [];
  let info: fs.Stats;
  try {
    info = fs.statSync(directoryPath);
  } catch (error) {
    return validationResult(
      [],
      [
        diagnostic(
          nodeErrnoCode(error) === 'ENOENT' && isSymbolicLink(directoryPath)
            ? 'skill-resource-broken-symlink'
            : 'skill-entrypoint-missing',
          'error',
          `Skill directory is not readable: ${directoryPath}`,
          directoryPath,
          source,
        ),
      ],
    );
  }
  if (!info.isDirectory()) {
    return validationResult(
      [],
      [
        diagnostic(
          'skill-entrypoint-missing',
          'error',
          'Skill path must be a directory containing SKILL.md',
          entrypoint,
          source,
        ),
      ],
    );
  }

  let markdown: string;
  try {
    markdown = fs.readFileSync(entrypoint, 'utf8');
  } catch (error) {
    const code = nodeErrnoCode(error);
    const broken = code === 'ENOENT' && isSymbolicLink(entrypoint);
    return validationResult(
      [],
      [
        diagnostic(
          broken ? 'skill-resource-broken-symlink' : 'skill-entrypoint-missing',
          'error',
          broken
            ? `Skill entrypoint is a broken symlink: ${entrypoint}`
            : `Skill directory is missing a readable SKILL.md entrypoint: ${entrypoint}`,
          entrypoint,
          source,
        ),
      ],
    );
  }

  const parsed = parseFrontMatter(markdown);
  if (parsed.error) {
    diagnostics.push(
      diagnostic('skill-frontmatter-invalid', 'error', parsed.error, entrypoint, source),
    );
  }
  const skill = parseSkillMarkdown(markdown, {
    basePath: directoryPath,
    fallbackName: basename(directoryPath),
  });
  diagnostics.push(
    ...validateSkillDefinition(skill, { path: entrypoint, source }).diagnostics,
    ...validateFrontMatterFields(parsed.yaml, entrypoint, source, skill.name),
    ...validateResources(directoryPath, markdown, source, skill.name),
  );
  return validationResult([skill], diagnostics);
}

/** Validate every discovered skill under one catalog directory. */
export function validateSkillsDirectory(directory: string): SkillValidationResult {
  return validateSkillCatalog({ directories: [directory], sourceMode: 'replace' });
}

/** Validate an already-resolved catalog in the exact source precedence supplied. */
export function validateResolvedSkillCatalog(
  resolvedSources: ResolvedSkillSources,
  options: ValidateResolvedSkillCatalogOptions = {},
): SkillValidationResult {
  const diagnostics = resolvedSources.diagnostics.map((item) =>
    diagnostic(
      item.code,
      item.severity,
      item.message,
      item.path,
      {
        path: item.path,
        origin: item.origin,
        precedence: item.precedence,
      },
      undefined,
      item.duplicateOf,
    ),
  );
  const skills: AgentSkill[] = [];
  const definitions = new Map<string, { skill: AgentSkill; source: ResolvedAgentSource }>();

  for (const source of resolvedSources.sources) {
    const sourceContext: SkillDiagnosticSource = source;
    let files: string[];
    try {
      files = walkSkillFiles(source.path, options);
    } catch {
      diagnostics.push(
        diagnostic(
          'source-unreadable',
          'error',
          `Source is unreadable: ${source.path}`,
          source.path,
          sourceContext,
        ),
      );
      continue;
    }
    for (const file of files) {
      const result = validateSkillDirectory(dirname(file));
      const sourceDiagnostics = result.diagnostics.map((item) => ({
        ...item,
        source: sourceContext,
      }));
      diagnostics.push(...sourceDiagnostics);
      const skill = result.skills[0];
      if (skill == null) continue;
      const kept = definitions.get(skill.name);
      if (kept != null) {
        const shadowed = kept.source.precedence !== source.precedence;
        diagnostics.push(
          diagnostic(
            shadowed ? 'skill-shadowed' : 'skill-duplicate',
            'warning',
            shadowed
              ? `Skill "${skill.name}" is shadowed by higher-precedence definition ${kept.skill.basePath}`
              : `Skill "${skill.name}" duplicates definition ${kept.skill.basePath}`,
            skill.basePath,
            sourceContext,
            skill.name,
            kept.skill.basePath,
          ),
        );
        continue;
      }
      definitions.set(skill.name, { skill, source });
      skills.push(skill);
    }
  }
  return validationResult(skills, diagnostics);
}

/** Resolve and validate a catalog using the same roots and precedence as runtime discovery. */
export function validateSkillCatalog(
  options: ResolveSkillSourcesOptions = {},
): SkillValidationResult {
  return validateResolvedSkillCatalog(resolveSkillSources(options));
}

function validateFrontMatterFields(
  yaml: YamlMap,
  path: string,
  source: SkillDiagnosticSource,
  skillName: string,
): SkillCatalogDiagnostic[] {
  const diagnostics: SkillCatalogDiagnostic[] = [];
  if (!Object.hasOwn(yaml, 'name')) {
    diagnostics.push(
      diagnostic('skill-name-invalid', 'error', 'name is required in frontmatter', path, source),
    );
  }
  if (Object.hasOwn(yaml, 'name') && typeof yaml.name !== 'string') {
    diagnostics.push(
      diagnostic(
        'skill-frontmatter-invalid',
        'error',
        'name must be a string',
        path,
        source,
        skillName,
      ),
    );
  }
  if (Object.hasOwn(yaml, 'description') && typeof yaml.description !== 'string') {
    diagnostics.push(
      diagnostic(
        'skill-frontmatter-invalid',
        'error',
        'description must be a string',
        path,
        source,
        skillName,
      ),
    );
  }
  const compatibility = yaml.compatibility;
  if (
    compatibility != null &&
    (typeof compatibility !== 'string' ||
      compatibility.trim().length === 0 ||
      compatibility.length > 500)
  ) {
    diagnostics.push(
      diagnostic(
        'skill-frontmatter-invalid',
        'error',
        'compatibility must be a string of at most 500 characters',
        path,
        source,
        skillName,
      ),
    );
  }
  if (
    yaml.metadata != null &&
    (typeof yaml.metadata !== 'object' || Array.isArray(yaml.metadata))
  ) {
    diagnostics.push(
      diagnostic(
        'skill-frontmatter-invalid',
        'error',
        'metadata must be a mapping',
        path,
        source,
        skillName,
      ),
    );
  }
  return diagnostics;
}

function validateResources(
  directory: string,
  markdown: string,
  source: SkillDiagnosticSource,
  skillName: string,
): SkillCatalogDiagnostic[] {
  const diagnostics: SkillCatalogDiagnostic[] = [];
  const checked = new Set<string>();
  for (const resource of discoverResources(directory)) {
    checked.add(resource);
    const finding = validateResource(resource, directory, source, skillName);
    if (finding) diagnostics.push(finding);
  }
  for (const reference of referencedResourcePaths(markdown)) {
    const resource = resolve(directory, reference);
    if (checked.has(resource)) continue;
    checked.add(resource);
    const finding = validateResource(resource, directory, source, skillName);
    if (finding) diagnostics.push(finding);
  }
  return diagnostics;
}

function validateResource(
  path: string,
  directory: string,
  source: SkillDiagnosticSource,
  skillName: string,
): SkillCatalogDiagnostic | undefined {
  if (!isContained(path, directory)) {
    return diagnostic(
      'skill-resource-outside-directory',
      'error',
      `Resource resolves outside the skill directory: ${path}`,
      path,
      source,
      skillName,
    );
  }
  try {
    const info = fs.lstatSync(path);
    let realPath: string;
    if (info.isSymbolicLink()) {
      try {
        fs.statSync(path);
        realPath = fs.realpathSync(path);
      } catch {
        return diagnostic(
          'skill-resource-broken-symlink',
          'error',
          `Resource is a broken symlink: ${path}`,
          path,
          source,
          skillName,
        );
      }
    } else {
      realPath = fs.realpathSync(path);
    }
    if (!isContained(realPath, fs.realpathSync(directory))) {
      return diagnostic(
        'skill-resource-outside-directory',
        'error',
        `Resource resolves outside the skill directory: ${path}`,
        path,
        source,
        skillName,
        realPath,
      );
    }
    fs.accessSync(path, constants.R_OK);
  } catch (error) {
    if (nodeErrnoCode(error) === 'ENOENT') {
      return diagnostic(
        'skill-resource-missing',
        'error',
        `Referenced resource does not exist: ${path}`,
        path,
        source,
        skillName,
      );
    }
    return diagnostic(
      'skill-resource-unreadable',
      'error',
      `Resource is unreadable: ${path}`,
      path,
      source,
      skillName,
    );
  }
  return undefined;
}

function discoverResources(directory: string): string[] {
  const resources: string[] = [];
  const stack = [directory];
  while (stack.length > 0) {
    const path = stack.pop();
    if (path == null) break;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(path, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      if (entry.name === 'SKILL.md' || SKIP_DIR_NAMES.has(entry.name)) continue;
      const child = join(path, entry.name);
      resources.push(child);
      if (entry.isDirectory()) stack.push(child);
    }
  }
  return resources;
}

function referencedResourcePaths(markdown: string): string[] {
  const references = new Set<string>();
  for (const match of markdown.matchAll(/\]\(<?([^\s)>]+)>?(?:\s+['"][^'"]*['"])?\)/g)) {
    addResourceReference(references, match[1]);
  }
  for (const match of markdown.matchAll(
    /(?:^|[\s`'"(])((?:scripts|references|assets)\/[\w@+.,/=-]+)/gm,
  )) {
    addResourceReference(references, match[1]);
  }
  return [...references];
}

function addResourceReference(references: Set<string>, raw: string | undefined): void {
  if (!raw) return;
  const value = decodeURIComponentSafe(raw.split(/[?#]/, 1)[0] ?? '').replace(/^\.\//, '');
  const normalized = value.replace(/[.,;:]+$/, '');
  if (
    !normalized ||
    normalized.startsWith('#') ||
    isAbsolute(normalized) ||
    /^[a-z][a-z\d+.-]*:/i.test(normalized)
  ) {
    return;
  }
  if (
    normalized.startsWith('../') ||
    RESOURCE_DIRECTORY_NAMES.has(normalized.split('/')[0] ?? '')
  ) {
    references.add(normalized);
  }
}

function parseFrontMatter(markdown: string): { yaml: YamlMap; error?: string } {
  const match = /^---[ \t]*\r?\n([\s\S]*?)\r?\n---[ \t]*(?:\r?\n|$)/.exec(markdown);
  if (match == null) return { yaml: {}, error: 'SKILL.md must begin with valid YAML frontmatter' };
  try {
    const yaml = parseYaml(match[1] ?? '');
    if (yaml == null || typeof yaml !== 'object' || Array.isArray(yaml)) {
      return { yaml: {}, error: 'SKILL.md frontmatter must be a YAML mapping' };
    }
    return { yaml };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { yaml: {}, error: `SKILL.md frontmatter is invalid: ${message}` };
  }
}

function walkSkillFiles(root: string, options: ValidateResolvedSkillCatalogOptions): string[] {
  const out: string[] = [];
  const stack = [root];
  while (stack.length > 0) {
    const directory = stack.pop();
    if (directory == null) break;
    if (isPolicySuppressed(directory, 'directory', options)) continue;
    const entries = fs
      .readdirSync(directory, { withFileTypes: true })
      .sort((left, right) => left.name.localeCompare(right.name));
    for (const entry of entries) {
      if (SKIP_DIR_NAMES.has(entry.name)) continue;
      const path = join(directory, entry.name);
      if (entry.isDirectory()) stack.push(path);
      else if (
        entry.isFile() &&
        entry.name === 'SKILL.md' &&
        !isPolicySuppressed(path, 'file', options)
      ) {
        out.push(path);
      }
    }
  }
  return out;
}

function isPolicySuppressed(
  path: string,
  kind: 'file' | 'directory',
  options: ValidateResolvedSkillCatalogOptions,
): boolean {
  if (options.aiIgnorePolicy == null) return false;
  const evaluation = evaluateAiIgnorePath(options.aiIgnorePolicy, path, { kind });
  if (!evaluation.ignored) return false;
  options.onSuppressed?.(aiIgnoreSuppressionDiagnostic(evaluation, 'skill-discovery', kind));
  return true;
}

function validationResult(
  skills: readonly AgentSkill[],
  diagnostics: readonly SkillCatalogDiagnostic[],
): SkillValidationResult {
  return {
    valid: diagnostics.every((item) => item.severity !== 'error'),
    skills,
    diagnostics,
  };
}

function diagnostic(
  code: SkillCatalogDiagnosticCode,
  severity: 'error' | 'warning',
  message: string,
  path: string,
  source: SkillDiagnosticSource,
  skillName?: string,
  relatedPath?: string,
): SkillCatalogDiagnostic {
  return { code, severity, message, path, source, skillName, relatedPath };
}

function resolveDiagnosticPath(path: string): string {
  return path === '.' ? path : resolve(path);
}

function isContained(path: string, root: string): boolean {
  if (path === root) return true;
  const rel = relative(root, path);
  return rel !== '' && rel !== '..' && !rel.startsWith(`..${sep}`) && !isAbsolute(rel);
}

function isSymbolicLink(path: string): boolean {
  try {
    return fs.lstatSync(path).isSymbolicLink();
  } catch {
    return false;
  }
}

function decodeURIComponentSafe(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}
