/**
 * Thin decorator DX for Agent Skills. Decorators only store metadata; apply
 * helpers translate into the existing builders. Builders remain the escape hatch.
 */

import type { ChatModel } from '@di-framework/ai';
import { defineMetadata, getOwnMetadata } from '@di-framework/core/container';
import {
  type CreateSkillsAgentOptions,
  SkillsAgent,
  type SkillsAgentBuilder,
} from './create-skills-agent.ts';
import {
  type AgentSkill,
  type AgentSkillCreateOptions,
  agentSkill,
} from './parse-skill-markdown.ts';
import type { SkillSourceMode } from './resolve-skill-sources.ts';
import type { SkillsFluent } from './skills-fluent.ts';
import {
  type BuildSkillsIndexOptions,
  SkillsIndex,
  type SkillsIndexBuilder,
} from './skills-index.ts';
import {
  SkillsToolbox,
  type SkillsToolboxBuilder,
  type SkillsToolboxOptions,
} from './skills-toolbox.ts';

const SKILLS_CATALOG_KEY = 'ai-utils:skills';
const SKILLS_SEMANTIC_KEY = 'ai-utils:skills-semantic';
const SKILLS_INDEX_KEY = 'ai-utils:skills-index';
const SKILLS_DECLARED_KEY = 'ai-utils:skills-declared';

type AnyConstructor = new (...args: never[]) => object;
type DecoratedTarget = object | AnyConstructor;

/** Catalog sources declared with {@link Skills}. */
export interface SkillsDecoratorOptions {
  readonly directories?: readonly string[];
  readonly packages?: readonly string[];
  readonly files?: readonly string[];
  readonly skills?: readonly AgentSkill[];
  readonly workspace?: string;
  readonly userDirectory?: string;
  readonly sourceMode?: SkillSourceMode;
}

/**
 * Serializable semantic retrieval settings for {@link SemanticSkillDiscovery}.
 * Custom embedders and stores stay on apply-helper overrides / builders.
 */
export interface SemanticSkillDiscoveryDecoratorOptions {
  readonly indexFile?: string;
  readonly limit?: number;
  readonly minScore?: number;
  readonly recentUserMessages?: number;
  readonly namespace?: string;
  readonly timeoutMs?: number;
}

/** Build-time index declaration for {@link SkillsIndexConfig}. Does not run indexing. */
export interface SkillsIndexConfigOptions {
  readonly directories?: readonly string[];
  readonly files?: readonly string[];
  readonly skills?: readonly AgentSkill[];
  readonly outputFile?: string;
  readonly threshold?: number;
  readonly retrievalLimit?: number;
  readonly chunkTokens?: number;
  readonly chunkOverlapTokens?: number;
  readonly batchSize?: number;
}

/** Programmatic skill declaration. {@link content} defaults to an empty body. */
export type SkillDecoratorOptions = Omit<AgentSkillCreateOptions, 'content'> & {
  readonly content?: string;
};

export type SkillsToolboxFromOverrides = SkillsToolboxOptions;
export type SkillsAgentFromOverrides = CreateSkillsAgentOptions;
export type SkillsIndexFromOverrides = BuildSkillsIndexOptions;

function getCtor(target: DecoratedTarget): AnyConstructor {
  if (typeof target === 'function') return target as AnyConstructor;
  return (target as { constructor: AnyConstructor }).constructor;
}

/**
 * Declares skill catalog sources on a class. Prefer builders for most apps;
 * use {@link skillsToolboxBuilderFrom} / {@link skillsAgentBuilderFrom} to apply.
 */
export function Skills(options: SkillsDecoratorOptions = {}): ClassDecorator {
  return (target) => {
    defineMetadata(SKILLS_CATALOG_KEY, { ...options }, target);
    return target as never;
  };
}

/**
 * Attaches serializable semantic discovery settings. Runtime embedder / store
 * wiring stays on apply helpers and builders.
 */
export function SemanticSkillDiscovery(
  options: SemanticSkillDiscoveryDecoratorOptions = {},
): ClassDecorator {
  return (target) => {
    defineMetadata(SKILLS_SEMANTIC_KEY, { ...options }, target);
    return target as never;
  };
}

/**
 * Declares build-time skills index configuration. Does not build the index;
 * call {@link skillsIndexBuilderFrom} then `.build()`, or use `di-framework skills index build`.
 */
export function SkillsIndexConfig(options: SkillsIndexConfigOptions = {}): ClassDecorator {
  return (target) => {
    defineMetadata(SKILLS_INDEX_KEY, { ...options }, target);
    return target as never;
  };
}

/**
 * Declares an in-memory {@link AgentSkill} on a class. Filesystem `SKILL.md`
 * remains the interoperable source format.
 */
export function Skill(options: SkillDecoratorOptions): ClassDecorator {
  return (target) => {
    const skill = agentSkill({
      ...options,
      content: options.content ?? '',
    });
    const ctor = getCtor(target);
    const existing = (getOwnMetadata(SKILLS_DECLARED_KEY, ctor) as AgentSkill[] | undefined) ?? [];
    defineMetadata(SKILLS_DECLARED_KEY, [...existing, skill], ctor);
    return target as never;
  };
}

export function getSkillsMetadata(target: DecoratedTarget): SkillsDecoratorOptions | undefined {
  return getOwnMetadata(SKILLS_CATALOG_KEY, getCtor(target)) as SkillsDecoratorOptions | undefined;
}

export function getSemanticSkillDiscoveryMetadata(
  target: DecoratedTarget,
): SemanticSkillDiscoveryDecoratorOptions | undefined {
  return getOwnMetadata(SKILLS_SEMANTIC_KEY, getCtor(target)) as
    | SemanticSkillDiscoveryDecoratorOptions
    | undefined;
}

export function getSkillsIndexMetadata(
  target: DecoratedTarget,
): SkillsIndexConfigOptions | undefined {
  return getOwnMetadata(SKILLS_INDEX_KEY, getCtor(target)) as SkillsIndexConfigOptions | undefined;
}

export function getDeclaredSkills(target: DecoratedTarget): readonly AgentSkill[] {
  return (getOwnMetadata(SKILLS_DECLARED_KEY, getCtor(target)) as AgentSkill[] | undefined) ?? [];
}

/**
 * Merge decorator metadata into {@link SkillsToolboxOptions}. Overrides win and
 * may supply non-serializable runtime deps (embedder, stores, chatModel).
 */
export function skillsToolboxOptionsFrom(
  target: DecoratedTarget,
  overrides: SkillsToolboxFromOverrides = {},
): SkillsToolboxOptions {
  const catalog = getSkillsMetadata(target) ?? {};
  const semantic = getSemanticSkillDiscoveryMetadata(target);
  const declared = getDeclaredSkills(target);

  const directories = overrides.directories ?? catalog.directories;

  const skills = [...(catalog.skills ?? []), ...declared, ...(overrides.skills ?? [])];

  let semanticDiscovery: SkillsToolboxOptions['semanticDiscovery'];
  if (overrides.semanticDiscovery !== undefined) {
    if (
      semantic != null &&
      overrides.semanticDiscovery != null &&
      typeof overrides.semanticDiscovery === 'object'
    ) {
      semanticDiscovery = { ...semantic, ...overrides.semanticDiscovery };
    } else {
      semanticDiscovery = overrides.semanticDiscovery;
    }
  } else if (semantic != null) {
    semanticDiscovery = { ...semantic };
  }

  const { skills: _catalogSkills, ...catalogRest } = catalog;

  return {
    ...catalogRest,
    ...overrides,
    directories,
    packages: overrides.packages ?? catalog.packages,
    files: overrides.files ?? catalog.files,
    workspace: overrides.workspace ?? catalog.workspace,
    userDirectory: overrides.userDirectory ?? catalog.userDirectory,
    sourceMode: overrides.sourceMode ?? catalog.sourceMode,
    skills: skills.length > 0 ? skills : undefined,
    semanticDiscovery,
  };
}

/** Prefill {@link SkillsToolbox.builder} from decorator metadata. */
export function skillsToolboxBuilderFrom(
  target: DecoratedTarget,
  overrides: SkillsToolboxFromOverrides = {},
): SkillsToolboxBuilder {
  return applyToolboxOptions(SkillsToolbox.builder(), skillsToolboxOptionsFrom(target, overrides));
}

/** Build a toolbox from decorator metadata (advisor included when configured). */
export function skillsToolboxFrom(
  target: DecoratedTarget,
  overrides: SkillsToolboxFromOverrides = {},
) {
  return skillsToolboxBuilderFrom(target, overrides).build();
}

/** Prefill {@link SkillsAgent.builder} from decorator metadata. */
export function skillsAgentBuilderFrom(
  target: DecoratedTarget,
  overrides: SkillsAgentFromOverrides = {},
): SkillsAgentBuilder {
  const toolboxOptions = skillsToolboxOptionsFrom(target, overrides);
  const builder = applyToolboxOptions(SkillsAgent.builder(), toolboxOptions);
  if (overrides.system != null) builder.system(overrides.system);
  if (overrides.extraTools?.length) builder.extraTools(...overrides.extraTools);
  if (overrides.chatClient != null) builder.chatClient(overrides.chatClient);
  if (overrides.defaultOptions != null) builder.defaultOptions(overrides.defaultOptions);
  if (overrides.advisors?.length) builder.advisors(...overrides.advisors);
  if (overrides.conversationMemory != null) {
    builder.conversationMemory(overrides.conversationMemory);
  }
  if (overrides.defaultConversationId != null) {
    builder.defaultConversationId(overrides.defaultConversationId);
  }
  if (overrides.builder != null) builder.clientBuilderOptions(overrides.builder);
  if (overrides.instructionDiscovery !== undefined) {
    builder.instructionDiscovery(overrides.instructionDiscovery);
  }
  return builder;
}

/** Build a chat agent from decorator metadata. */
export function skillsAgentFrom(target: DecoratedTarget, overrides: SkillsAgentFromOverrides = {}) {
  return skillsAgentBuilderFrom(target, overrides).build();
}

/** Prefill {@link SkillsIndex.builder} from {@link SkillsIndexConfig} metadata. */
export function skillsIndexBuilderFrom(
  target: DecoratedTarget,
  overrides: SkillsIndexFromOverrides = {},
): SkillsIndexBuilder {
  const meta = getSkillsIndexMetadata(target) ?? {};
  const declared = getDeclaredSkills(target);
  const catalog = getSkillsMetadata(target);
  const skills = [
    ...(meta.skills ?? []),
    ...(catalog?.skills ?? []),
    ...declared,
    ...(overrides.skills ?? []),
  ];
  const options: BuildSkillsIndexOptions = {
    directories: overrides.directories ?? meta.directories ?? catalog?.directories,
    files: overrides.files ?? meta.files ?? catalog?.files,
    skills: skills.length > 0 ? skills : undefined,
    outputFile: overrides.outputFile ?? meta.outputFile,
    threshold: overrides.threshold ?? meta.threshold,
    retrievalLimit: overrides.retrievalLimit ?? meta.retrievalLimit,
    chunkTokens: overrides.chunkTokens ?? meta.chunkTokens,
    chunkOverlapTokens: overrides.chunkOverlapTokens ?? meta.chunkOverlapTokens,
    batchSize: overrides.batchSize ?? meta.batchSize,
    embedder: overrides.embedder,
    writer: overrides.writer,
    force: overrides.force,
    onProgress: overrides.onProgress,
  };
  return applyIndexOptions(SkillsIndex.builder(), options);
}

function applyToolboxOptions<T extends SkillsFluent<T>>(
  builder: T,
  options: SkillsToolboxOptions,
): T {
  let b = builder;
  if (options.directories != null) {
    b = b.addSkillsDirectories(options.directories);
  }
  if (options.files?.length) {
    for (const file of options.files) b = b.addSkillsFile(file);
  }
  if (options.packages?.length) b = b.addPackages(options.packages);
  if (options.skills?.length) b = b.addSkills(options.skills);
  if (options.workspace != null) b = b.workspace(options.workspace);
  if (options.userDirectory != null) b = b.userDirectory(options.userDirectory);
  if (options.sourceMode != null) b = b.sourceMode(options.sourceMode);
  if (options.extraAllowedDirectories?.length) {
    b = b.extraAllowedDirectories(options.extraAllowedDirectories);
  }
  if (options.toolName != null) b = b.toolName(options.toolName);
  if (options.toolDescriptionTemplate != null) {
    b = b.toolDescriptionTemplate(options.toolDescriptionTemplate);
  }
  if (options.onActivate != null) b = b.onActivate(options.onActivate);
  if (options.glob != null) b = b.glob(options.glob);
  if (options.grep != null) b = b.grep(options.grep);
  if (options.list != null) b = b.list(options.list);
  if (options.write != null) b = b.write(options.write);
  if (options.aiIgnore != null) b = b.aiIgnore(options.aiIgnore);
  if (options.shell != null) b = b.shell(options.shell);
  if (options.shellTimeoutMs != null) b = b.shellTimeoutMs(options.shellTimeoutMs);
  if (options.confirmShell != null) b = b.confirmShell(options.confirmShell);
  if (options.todos != null) b = b.todos(options.todos);
  if (options.askUser != null) b = b.askUser(options.askUser);
  if (options.web != null) b = b.web(options.web);
  if (options.memories != null) b = b.memories(options.memories);
  if (options.task != null) b = b.task(options.task);
  if (options.chatModel != null) b = b.chatModel(options.chatModel as ChatModel);
  if (options.perSkillSandbox != null) b = b.perSkillSandbox(options.perSkillSandbox);
  if (options.semanticDiscovery != null) b = b.semanticDiscovery(options.semanticDiscovery);
  return b;
}

function applyIndexOptions(
  builder: SkillsIndexBuilder,
  options: BuildSkillsIndexOptions,
): SkillsIndexBuilder {
  let b = builder;
  if (options.directories?.length) b = b.addSkillsDirectories(options.directories);
  if (options.files?.length) b = b.addSkillsFiles(options.files);
  if (options.skills?.length) b = b.addSkills(options.skills);
  if (options.outputFile != null) b = b.outputFile(options.outputFile);
  if (options.threshold != null) b = b.threshold(options.threshold);
  if (options.retrievalLimit != null) b = b.retrievalLimit(options.retrievalLimit);
  if (options.batchSize != null) b = b.batchSize(options.batchSize);
  if (options.chunkTokens != null) b = b.chunkTokens(options.chunkTokens);
  if (options.chunkOverlapTokens != null) b = b.chunkOverlapTokens(options.chunkOverlapTokens);
  if (options.embedder != null) b = b.embedder(options.embedder);
  if (options.writer != null) b = b.writer(options.writer);
  if (options.force != null) b = b.force(options.force);
  if (options.onProgress != null) b = b.onProgress(options.onProgress);
  return b;
}
