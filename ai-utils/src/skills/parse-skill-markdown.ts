/**
 * SKILL.md front matter + body, aligned with Spring AI {@code MarkdownParser}
 * and the Agent Skills spec (name + description at minimum).
 */

import {
  flattenYamlMap,
  parseYamlMap,
  type YamlMap,
  yamlValueToString,
} from '../yaml/parse-yaml.ts';

export interface AgentSkill {
  readonly name: string;
  readonly description?: string;
  readonly basePath: string;
  readonly frontMatter: Readonly<Record<string, string>>;
  readonly yaml: Readonly<YamlMap>;
  /** Complete SKILL.md source used by build-time semantic indexing. */
  readonly source: string;
  readonly content: string;
  /** Parsed from {@code allowed-tools} front matter when present. */
  readonly allowedTools?: readonly string[];
  readonly license?: string;
  readonly compatibility?: string;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface ParseSkillMarkdownOptions {
  readonly basePath?: string;
  readonly fallbackName?: string;
}

export interface AgentSkillCreateOptions {
  readonly name: string;
  readonly description?: string;
  readonly content: string;
  readonly basePath?: string;
  readonly frontMatter?: Readonly<Record<string, string>>;
  readonly allowedTools?: readonly string[];
  readonly license?: string;
  readonly compatibility?: string;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

/**
 * Build an in-memory skill (no filesystem).
 */
export function agentSkill(options: AgentSkillCreateOptions): AgentSkill {
  const name = options.name.trim();
  if (!name) {
    throw new Error('Skill name is required');
  }
  const frontMatter: Record<string, string> = {
    ...options.frontMatter,
    name,
  };
  if (options.description != null) {
    frontMatter.description = options.description;
  }
  if (options.allowedTools) {
    frontMatter['allowed-tools'] = options.allowedTools.join(', ');
  }
  if (options.license) frontMatter.license = options.license;
  if (options.compatibility) frontMatter.compatibility = options.compatibility;
  return {
    name,
    description: options.description ?? options.frontMatter?.description,
    basePath: options.basePath ?? '.',
    frontMatter,
    yaml: { ...frontMatter },
    source: serializeSkillMarkdown(frontMatter, options.content),
    content: options.content,
    allowedTools: options.allowedTools ?? parseAllowedTools(frontMatter['allowed-tools']),
    license: options.license ?? frontMatter.license,
    compatibility: options.compatibility ?? frontMatter.compatibility,
    metadata: options.metadata,
  };
}

/**
 * Parse a SKILL.md document. Front matter is YAML (maps, lists, scalars, blocks).
 */
export function parseSkillMarkdown(
  markdown: string,
  options: ParseSkillMarkdownOptions = {},
): AgentSkill {
  const { yaml, content } = splitFrontMatter(markdown ?? '');
  const name = (yamlValueToString(yaml.name) ?? options.fallbackName ?? '').trim();
  if (!name) {
    throw new Error('SKILL.md is missing a name (front matter or folder name)');
  }
  const frontMatter: Record<string, string> = { ...flattenYamlMap(yaml), name };
  const metadataRaw = yaml.metadata;
  const metadata =
    metadataRaw != null && typeof metadataRaw === 'object' && !Array.isArray(metadataRaw)
      ? (metadataRaw as Record<string, unknown>)
      : undefined;
  return {
    name,
    description: yamlValueToString(yaml.description) ?? frontMatter.description,
    basePath: options.basePath ?? '.',
    frontMatter,
    yaml,
    source: markdown,
    content,
    allowedTools: parseAllowedTools(yaml['allowed-tools'] ?? frontMatter['allowed-tools']),
    license: yamlValueToString(yaml.license),
    compatibility: yamlValueToString(yaml.compatibility),
    metadata,
  };
}

function serializeSkillMarkdown(frontMatter: Readonly<Record<string, string>>, content: string) {
  const fields = Object.entries(frontMatter).map(
    ([key, value]) => `${key}: ${JSON.stringify(value)}`,
  );
  return `---\n${fields.join('\n')}\n---\n\n${content}`;
}

export function parseAllowedTools(value: unknown): readonly string[] | undefined {
  if (value == null) return undefined;
  if (Array.isArray(value)) {
    const tools = value
      .map((item) => (typeof item === 'string' ? item.trim() : yamlValueToString(item as never)))
      .filter((item): item is string => Boolean(item && item.length > 0));
    return tools.length > 0 ? tools : undefined;
  }
  if (typeof value !== 'string' || !value.trim()) return undefined;
  const tools = value
    .split(',')
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
  return tools.length > 0 ? tools : undefined;
}

function splitFrontMatter(markdown: string): {
  yaml: YamlMap;
  content: string;
} {
  if (!markdown.startsWith('---')) {
    return { yaml: {}, content: markdown };
  }

  const endIndex = markdown.indexOf('---', 3);
  if (endIndex === -1) {
    return { yaml: {}, content: markdown };
  }

  const section = markdown.slice(3, endIndex).replace(/^\n/, '');
  const content = markdown.slice(endIndex + 3).trim();
  try {
    return { yaml: parseYamlMap(section), content };
  } catch {
    return { yaml: {}, content: markdown };
  }
}
