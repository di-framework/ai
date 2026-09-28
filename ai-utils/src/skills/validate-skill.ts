import { basename } from 'node:path';
import type { AgentSkill } from './parse-skill-markdown.ts';

const NAME_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export interface ValidateSkillOptions {
  /**
   * When true (default for files loaded from disk), {@link AgentSkill.name}
   * must equal the skill directory basename.
   */
  readonly matchDirectoryName?: boolean;
}

/**
 * Enforce agentskills.io {@code name} and {@code description} rules.
 */
export function validateSkill(skill: AgentSkill, options: ValidateSkillOptions = {}): void {
  const nameError = validateSkillName(skill.name);
  if (nameError) {
    throw new Error(formatSkillError(skill, nameError));
  }

  const descriptionError = validateSkillDescription(skill.description);
  if (descriptionError) {
    throw new Error(formatSkillError(skill, descriptionError));
  }

  if (options.matchDirectoryName !== false && skill.basePath && skill.basePath !== '.') {
    const folder = basename(skill.basePath);
    if (folder !== skill.name) {
      throw new Error(
        formatSkillError(
          skill,
          `name "${skill.name}" must match the skill directory name "${folder}"`,
        ),
      );
    }
  }
}

export function validateSkillName(name: string | undefined): string | undefined {
  const value = name?.trim() ?? '';
  if (!value) return 'name is required';
  if (value.length > 64) return 'name must be at most 64 characters';
  if (!NAME_PATTERN.test(value)) {
    return 'name must be lowercase letters, numbers, and single hyphens (no leading, trailing, or consecutive hyphens)';
  }
  return undefined;
}

export function validateSkillDescription(description: string | undefined): string | undefined {
  const value = description?.trim() ?? '';
  if (!value) return 'description is required';
  if (value.length > 1024) return 'description must be at most 1024 characters';
  return undefined;
}

function formatSkillError(skill: AgentSkill, message: string): string {
  const where = skill.basePath && skill.basePath !== '.' ? ` (${skill.basePath})` : '';
  return `Invalid skill${where}: ${message}`;
}
