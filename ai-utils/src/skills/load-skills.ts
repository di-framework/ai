import { readFileSync, statSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import type { AiIgnorePolicy, AiIgnoreSuppressionDiagnostic } from '../policy/index.ts';
import { nodeErrnoCode } from '../sandbox/fs-error.ts';
import { expandUserPath } from '../sandbox/paths.ts';
import { walkFiles } from '../tools/walk-files.ts';
import { type AgentSkill, parseSkillMarkdown } from './parse-skill-markdown.ts';

/** Vendor-neutral workspace and user locations used by merge discovery. */
export const DEFAULT_SKILL_DIRECTORY_CANDIDATES = ['.agents/skills', '~/.agents/skills'] as const;

export interface LoadSkillsDirectoryOptions {
  readonly aiIgnorePolicy?: AiIgnorePolicy;
  readonly onSuppressed?: (diagnostic: AiIgnoreSuppressionDiagnostic) => void;
}

/**
 * Recursively load every {@code SKILL.md} under {@code rootDirectory}.
 */
export function loadSkillsDirectory(
  rootDirectory: string,
  options: LoadSkillsDirectoryOptions = {},
): AgentSkill[] {
  const rootPath = resolve(rootDirectory);
  let isDirectory = false;
  try {
    isDirectory = statSync(rootPath).isDirectory();
  } catch {
    throw new Error(`Root directory does not exist: ${rootDirectory}`);
  }
  if (!isDirectory) {
    throw new Error(`Path is not a directory: ${rootDirectory}`);
  }

  const skills: AgentSkill[] = [];
  for (const file of walkSkillFiles(rootPath, options)) {
    skills.push(loadSkillFile(file));
  }
  return skills;
}

/**
 * Load {@code SKILL.md} files from each root directory.
 */
export function loadSkillsDirectories(
  rootDirectories: readonly string[],
  options: LoadSkillsDirectoryOptions = {},
): AgentSkill[] {
  const skills: AgentSkill[] = [];
  for (const root of rootDirectories) {
    skills.push(...loadSkillsDirectory(root, options));
  }
  return skills;
}

/**
 * Candidates that exist as directories. Missing paths are skipped so default
 * {@code ~/.agents/skills} does not fail closed on a fresh machine.
 */
export function existingSkillDirectories(
  candidates: readonly string[] = DEFAULT_SKILL_DIRECTORY_CANDIDATES,
): string[] {
  const out: string[] = [];
  for (const candidate of candidates) {
    const resolved = resolve(expandUserPath(candidate));
    try {
      if (statSync(resolved).isDirectory()) {
        out.push(resolved);
      }
    } catch {
      // Missing or unreadable candidates are skipped.
    }
  }
  return out.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

/**
 * Parse a single {@code SKILL.md} file. {@link AgentSkill.basePath} is the
 * parent directory (the skill's execution environment).
 */
export function loadSkillFile(skillMdPath: string): AgentSkill {
  const absolute = resolve(skillMdPath);
  let markdown: string;
  try {
    markdown = readFileSync(absolute, 'utf8');
  } catch (error) {
    if (nodeErrnoCode(error) === 'EISDIR') {
      throw new Error(`Path is not a file: ${skillMdPath}`);
    }
    throw new Error(`SKILL.md does not exist: ${skillMdPath}`);
  }
  const basePath = dirname(absolute);
  return parseSkillMarkdown(markdown, {
    basePath,
    fallbackName: basename(basePath),
  });
}

function walkSkillFiles(root: string, options: LoadSkillsDirectoryOptions): string[] {
  const out: string[] = [];
  walkFiles(
    root,
    0,
    Number.MAX_SAFE_INTEGER,
    (file) => {
      if (basename(file) === 'SKILL.md') out.push(file);
    },
    {
      aiIgnorePolicy: options.aiIgnorePolicy,
      surface: 'skill-discovery',
      onSuppressed: (diagnostic) => {
        if (diagnostic.kind === 'directory' || basename(diagnostic.path) === 'SKILL.md') {
          options.onSuppressed?.(diagnostic);
        }
      },
    },
  );
  return out;
}
