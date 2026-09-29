import { uniqueResolvedRoots } from '../sandbox/paths.ts';
import type { AgentSkill } from './parse-skill-markdown.ts';

export interface SkillsRuntimeOptions {
  readonly workspace: string;
  readonly extraAllowedDirectories?: readonly string[];
  readonly skillDirectories: readonly string[];
  /** When true (default), file tools jail to the active skill after Skill runs. */
  readonly perSkillSandbox?: boolean;
}

export interface SkillsRuntime {
  activate(skill: AgentSkill): void;
  activeSkill(): AgentSkill | undefined;
  isToolAllowed(toolName: string): boolean;
  deniedToolMessage(toolName: string): string;
  fileDirectories(): readonly string[];
}

const ALWAYS_ALLOWED = new Set(['Skill']);

export function createSkillsRuntime(options: SkillsRuntimeOptions): SkillsRuntime {
  let active: AgentSkill | undefined;
  const baseDirs = uniqueResolvedRoots([
    options.workspace,
    ...options.skillDirectories,
    ...(options.extraAllowedDirectories ?? []),
  ]);
  const perSkill = options.perSkillSandbox !== false;

  return {
    activate(skill) {
      active = skill;
    },
    activeSkill() {
      return active;
    },
    isToolAllowed(toolName) {
      if (ALWAYS_ALLOWED.has(toolName)) return true;
      const allowed = active?.allowedTools;
      if (allowed == null || allowed.length === 0) return true;
      return allowed.some((name) => name === toolName);
    },
    deniedToolMessage(toolName) {
      const allowed = active?.allowedTools ?? [];
      return `Error: Tool ${toolName} is not in this skill's allowed-tools (${allowed.join(', ') || 'none'})`;
    },
    fileDirectories() {
      if (!perSkill || active == null || active.basePath === '.') {
        return baseDirs;
      }
      return uniqueResolvedRoots([
        options.workspace,
        active.basePath,
        ...(options.extraAllowedDirectories ?? []),
      ]);
    },
  };
}
