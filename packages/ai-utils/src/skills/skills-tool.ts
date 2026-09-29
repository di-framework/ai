import { resolve } from 'node:path';
import { functionToolCallback, type ToolCallback } from '@di-framework/ai';
import {
  type AiIgnorePolicy,
  type AiIgnoreSuppressionDiagnostic,
  aiIgnoreSuppressionDiagnostic,
  evaluateAiIgnorePath,
} from '../policy/index.ts';
import { loadSkillFile, loadSkillsDirectories } from './load-skills.ts';
import type { AgentSkill } from './parse-skill-markdown.ts';
import {
  runSkillAdapterOperation,
  SkillAdapterError,
  type SkillCatalogStore,
  type SkillDescriptor,
} from './skill-adapters.ts';

export const DEFAULT_SKILL_TOOL_NAME = 'Skill';

const DEFAULT_TOOL_DESCRIPTION_TEMPLATE = `Execute a skill within the main conversation

<skills_instructions>
When users ask you to perform tasks, check if any of the available skills below can help complete the task more effectively. Skills provide specialized capabilities and domain knowledge.

How to use skills:
- Invoke skills using this tool with the skill name only (no arguments)
- When you invoke a skill, you will see <command-message>The "{name}" skill is loading</command-message>
- The skill's prompt will expand and provide detailed instructions on how to complete the task

NOTE: Response always starts start with the base directory of the skill execution environment. You can use this to retrieve additional files of call shell commands.
Skill description follows after the base directory line.

Important:
- Only use skills listed in <available_skills> below
- Do not invoke a skill that is already running
</skills_instructions>

<available_skills>
%s
</available_skills>
`;

export interface SkillsInput {
  readonly command?: string;
}

export interface SkillsToolOptions {
  readonly skills?: readonly AgentSkill[];
  readonly directories?: readonly string[];
  readonly files?: readonly string[];
  readonly toolName?: string;
  readonly toolDescriptionTemplate?: string;
  readonly onActivate?: (skill: AgentSkill) => void;
  readonly onDuplicate?: (diagnostic: SkillDuplicateDiagnostic) => void;
  readonly aiIgnorePolicy?: AiIgnorePolicy;
  readonly onSuppressed?: (diagnostic: AiIgnoreSuppressionDiagnostic) => void;
}

export interface SkillDuplicateDiagnostic {
  readonly code: 'skill-duplicate';
  readonly severity: 'warning';
  readonly name: string;
  readonly keptSource: string;
  readonly ignoredSource: string;
}

export interface AsyncSkillsToolOptions
  extends Omit<SkillsToolOptions, 'skills' | 'directories' | 'files'> {
  readonly descriptors: readonly SkillDescriptor[];
  readonly catalogStore: SkillCatalogStore;
  readonly namespace?: string;
  readonly timeoutMs?: number;
}

/**
 * Collect skills from in-memory records, directories, and SKILL.md files.
 * Sources are ordered by precedence. The first entry with a name wins.
 */
export function collectSkills(options: SkillsToolOptions): AgentSkill[] {
  const collected: AgentSkill[] = [...(options.skills ?? [])];
  if (options.files?.length) {
    for (const file of options.files) {
      const path = resolve(file);
      const evaluation = options.aiIgnorePolicy
        ? evaluateAiIgnorePath(options.aiIgnorePolicy, path, { kind: 'file' })
        : undefined;
      if (evaluation?.ignored) {
        options.onSuppressed?.(
          aiIgnoreSuppressionDiagnostic(evaluation, 'skill-discovery', 'file'),
        );
        continue;
      }
      collected.push(loadSkillFile(path));
    }
  }
  if (options.directories?.length) {
    collected.push(
      ...loadSkillsDirectories(options.directories, {
        aiIgnorePolicy: options.aiIgnorePolicy,
        onSuppressed: options.onSuppressed,
      }),
    );
  }
  return [...toSkillsMap(collected, options.onDuplicate).values()];
}

export function skillToXml(skill: Pick<AgentSkill, 'name' | 'description'>): string {
  const entries = [
    ['name', skill.name] as const,
    ...(skill.description == null ? [] : ([['description', skill.description]] as const)),
  ]
    .map(([key, value]) => `  <${key}>${escapeXml(value)}</${key}>`)
    .join('\n');
  return `<skill>\n${entries}\n</skill>`;
}

export function formatSkillLoadResult(skill: AgentSkill): string {
  return `Base directory for this skill: ${skill.basePath}\n\n${skill.content}`;
}

export function formatSkillNotFound(command: string): string {
  return `Skill not found: ${command}`;
}

/**
 * Tool-based Agent Skills (agentskills.io / Spring {@code SkillsTool}).
 *
 * Discovery embeds name + description in the tool description. Activation
 * loads the full SKILL.md body plus the skill base directory.
 */
export function skillsTool(options: SkillsToolOptions): ToolCallback {
  const skills = collectSkills(options);
  if (skills.length === 0) {
    throw new Error('At least one skill must be configured');
  }

  const skillsMap = toSkillsMap(skills);
  const skillsXml = [...skillsMap.values()].map(skillToXml).join('\n');
  const template = options.toolDescriptionTemplate ?? DEFAULT_TOOL_DESCRIPTION_TEMPLATE;
  const description = template.includes('%s')
    ? template.replace('%s', skillsXml)
    : `${template}\n${skillsXml}`;

  return functionToolCallback<SkillsInput, string>({
    name: options.toolName ?? DEFAULT_SKILL_TOOL_NAME,
    description,
    inputSchema: {
      type: 'object',
      properties: {
        command: {
          type: 'string',
          description: 'The skill name (no arguments). E.g., "pdf" or "xlsx"',
        },
      },
      required: ['command'],
    },
    call: (input) => {
      const command = input?.command?.trim() ?? '';
      const skill = command ? skillsMap.get(command) : undefined;
      if (!skill) {
        return formatSkillNotFound(command);
      }
      options.onActivate?.(skill);
      return formatSkillLoadResult(skill);
    },
  });
}

/** Progressive-disclosure Skill tool that loads a body only after activation. */
export function asyncSkillsTool(options: AsyncSkillsToolOptions): ToolCallback {
  if (options.descriptors.length === 0) throw new Error('At least one skill must be configured');
  const descriptors = new Map(
    options.descriptors.map((descriptor) => [descriptor.name, descriptor]),
  );
  const skillsXml = [...descriptors.values()].map(descriptorToXml).join('\n');
  const template = options.toolDescriptionTemplate ?? DEFAULT_TOOL_DESCRIPTION_TEMPLATE;
  const description = template.includes('%s')
    ? template.replace('%s', skillsXml)
    : `${template}\n${skillsXml}`;

  return functionToolCallback<SkillsInput, string>({
    name: options.toolName ?? DEFAULT_SKILL_TOOL_NAME,
    description,
    inputSchema: {
      type: 'object',
      properties: {
        command: {
          type: 'string',
          description: 'The skill name (no arguments). E.g., "pdf" or "xlsx"',
        },
      },
      required: ['command'],
    },
    call: async (input) => {
      const command = input?.command?.trim() ?? '';
      const descriptor = command ? descriptors.get(command) : undefined;
      if (!descriptor) return formatSkillNotFound(command);
      const skill = await runSkillAdapterOperation(
        `Loading skill '${command}'`,
        () =>
          options.catalogStore.load(command, {
            namespace: options.namespace,
            expectedVersion: descriptor.version ?? descriptor.sourceHash,
          }),
        options.timeoutMs,
      );
      if (!skill) {
        throw new SkillAdapterError('MISSING_BODY', `Activated skill '${command}' has no body`);
      }
      if (skill.name !== descriptor.name || skill.description !== descriptor.description) {
        throw new SkillAdapterError('STALE_CATALOG', `Activated skill '${command}' is stale`);
      }
      options.onActivate?.(skill);
      return formatSkillLoadResult(skill);
    },
  });
}

export class SkillsToolBuilder {
  private readonly skills: AgentSkill[] = [];
  private directories: string[] = [];
  private files: string[] = [];
  private name = DEFAULT_SKILL_TOOL_NAME;
  private template?: string;
  private activate?: (skill: AgentSkill) => void;

  toolName(name: string): this {
    this.name = name;
    return this;
  }

  toolDescriptionTemplate(template: string): this {
    this.template = template;
    return this;
  }

  addSkill(skill: AgentSkill): this {
    this.skills.push(skill);
    return this;
  }

  addSkills(skills: readonly AgentSkill[]): this {
    this.skills.push(...skills);
    return this;
  }

  addSkillsDirectory(skillsRootDirectory: string): this {
    this.directories.push(skillsRootDirectory);
    return this;
  }

  addSkillsDirectories(skillsRootDirectories: readonly string[]): this {
    this.directories.push(...skillsRootDirectories);
    return this;
  }

  addSkillsFile(skillMdPath: string): this {
    this.files.push(skillMdPath);
    return this;
  }

  onActivate(handler: (skill: AgentSkill) => void): this {
    this.activate = handler;
    return this;
  }

  build(): ToolCallback {
    return skillsTool({
      skills: this.skills,
      directories: this.directories,
      files: this.files,
      toolName: this.name,
      toolDescriptionTemplate: this.template,
      onActivate: this.activate,
    });
  }
}

/**
 * Spring-style factory. Prefer {@link SkillsTool.builder}.
 */
export const SkillsTool = {
  builder(): SkillsToolBuilder {
    return new SkillsToolBuilder();
  },
  of(options: SkillsToolOptions): ToolCallback {
    return skillsTool(options);
  },
};

function toSkillsMap(
  skills: readonly AgentSkill[],
  onDuplicate?: (diagnostic: SkillDuplicateDiagnostic) => void,
): Map<string, AgentSkill> {
  const map = new Map<string, AgentSkill>();
  for (const skill of skills) {
    const kept = map.get(skill.name);
    if (kept) {
      onDuplicate?.({
        code: 'skill-duplicate',
        severity: 'warning',
        name: skill.name,
        keptSource: kept.basePath,
        ignoredSource: skill.basePath,
      });
      continue;
    }
    map.set(skill.name, skill);
  }
  return map;
}

function escapeXml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function descriptorToXml(descriptor: SkillDescriptor): string {
  return skillToXml({ name: descriptor.name, description: descriptor.description });
}
