import { existsSync } from 'node:fs';
import { isAbsolute, resolve } from 'node:path';
import type { ChatModel, ToolCallback } from '@di-framework/ai';
import {
  type AiIgnorePolicy,
  type AiIgnoreSuppressionDiagnostic,
  loadAiIgnorePolicy,
} from '../policy/index.ts';
import { expandUserPath, uniqueResolvedRoots } from '../sandbox/paths.ts';
import type { AgentSourceDiagnostic, ResolvedAgentSource } from '../sources/index.ts';
import type { AiIgnoreEnforcement, AiIgnoreToolPolicy } from '../tools/aiignore-enforcement.ts';
import { askUserQuestionTool, type QuestionHandler } from '../tools/ask-user-question-tool.ts';
import { type BashConfirmInput, bashTool } from '../tools/bash-tool.ts';
import { editTool } from '../tools/edit-tool.ts';
import { gateToolCallback } from '../tools/gate-tool.ts';
import { globTool } from '../tools/glob-tool.ts';
import { grepTool } from '../tools/grep-tool.ts';
import { listDirectoryTool } from '../tools/list-directory-tool.ts';
import { memoryTools } from '../tools/memory-tools.ts';
import { readTool } from '../tools/read-tool.ts';
import { taskTool } from '../tools/task-tool.ts';
import { todoWriteTool } from '../tools/todo-write-tool.ts';
import { webFetchTool } from '../tools/web-fetch-tool.ts';
import { webSearchTool } from '../tools/web-search-tool.ts';
import { writeTool } from '../tools/write-tool.ts';
import type { AgentSkill } from './parse-skill-markdown.ts';
import {
  type ResolvedSkillSources,
  resolveSkillSources,
  type SkillSourceMode,
} from './resolve-skill-sources.ts';
import {
  runSkillAdapterOperation,
  SkillAdapterError,
  type SkillCatalogStore,
  type SkillDescriptor,
  type SkillVectorSearch,
} from './skill-adapters.ts';
import type { SkillEmbedder } from './skill-embedder.ts';
import { SkillsFluent } from './skills-fluent.ts';
import { DEFAULT_SKILLS_INDEX_FILE, loadSkillsIndex } from './skills-index.ts';
import { SkillsRetrievalAdvisor } from './skills-retrieval-advisor.ts';
import { createSkillsRuntime, type SkillsRuntime } from './skills-runtime.ts';
import type { SkillDuplicateDiagnostic, SkillsToolOptions } from './skills-tool.ts';
import {
  asyncSkillsTool,
  collectSkills,
  DEFAULT_SKILL_TOOL_NAME,
  skillsTool,
} from './skills-tool.ts';
import { validateSkill, validateSkillDescription, validateSkillName } from './validate-skill.ts';

export interface SkillsToolboxWebOptions {
  readonly fetch?: boolean;
  readonly search?: boolean;
  readonly braveApiKey?: string;
}

export interface SkillsToolboxMemoriesOptions {
  readonly directory: string;
}

export interface SkillsToolboxTaskOptions {
  readonly chatModel: ChatModel;
  readonly system?: string;
}

export interface SkillsSemanticDiscoveryOptions {
  /** Generated JSONL path. Relative paths resolve from {@link workspace}. */
  readonly indexFile?: string;
  readonly limit?: number;
  readonly minScore?: number;
  readonly recentUserMessages?: number;
  readonly embedder?: SkillEmbedder;
  readonly catalogStore?: SkillCatalogStore;
  readonly vectorSearch?: SkillVectorSearch;
  readonly namespace?: string;
  readonly timeoutMs?: number;
}

export interface SkillsToolboxOptions extends SkillsToolOptions {
  readonly workspace?: string;
  /** Home root used for user-default discovery; defaults to the process home. */
  readonly userDirectory?: string;
  /** Merge explicit roots before defaults, or replace defaults entirely. */
  readonly sourceMode?: SkillSourceMode;
  readonly extraAllowedDirectories?: readonly string[];
  /** npm package names or paths that contain skill folders. */
  readonly packages?: readonly string[];
  readonly shell?: boolean;
  readonly shellTimeoutMs?: number;
  readonly confirmShell?: (input: BashConfirmInput) => boolean | Promise<boolean>;
  readonly glob?: boolean;
  readonly grep?: boolean;
  readonly list?: boolean;
  readonly write?: boolean;
  /** Opt in to root-workspace `.aiignore` enforcement for direct file tools. */
  readonly aiIgnore?: AiIgnoreEnforcement;
  readonly todos?: boolean;
  readonly askUser?: QuestionHandler;
  readonly web?: boolean | SkillsToolboxWebOptions;
  readonly memories?: boolean | SkillsToolboxMemoriesOptions;
  readonly task?: boolean | SkillsToolboxTaskOptions;
  readonly chatModel?: ChatModel;
  readonly perSkillSandbox?: boolean;
  /**
   * Retrieve a small Skill catalog from a build-time semantic index. When
   * omitted, the default index is used automatically if it exists.
   */
  readonly semanticDiscovery?: boolean | SkillsSemanticDiscoveryOptions;
}

export interface SkillsToolbox {
  readonly skills: readonly AgentSkill[];
  readonly descriptors: readonly SkillDescriptor[];
  readonly allowedDirectories: readonly string[];
  readonly tools: readonly ToolCallback[];
  readonly runtime: SkillsRuntime;
  readonly retrievalAdvisor?: SkillsRetrievalAdvisor;
  readonly skillSources: readonly ResolvedAgentSource[];
  readonly skillDiagnostics: readonly SkillSourceDiagnostic[];
}

export type SkillSourceDiagnostic =
  | AgentSourceDiagnostic
  | SkillDuplicateDiagnostic
  | AiIgnoreSuppressionDiagnostic;

/** Preferred factory: {@code SkillsToolbox.builder().addSkillsDirectory(...).build()}. */
export const SkillsToolbox = {
  builder(): SkillsToolboxBuilder {
    return new SkillsToolboxBuilder();
  },
  of(options: SkillsToolboxOptions = {}): SkillsToolbox {
    return createSkillsToolbox(options);
  },
  ofAsync(options: SkillsToolboxOptions = {}): Promise<SkillsToolbox> {
    return createSkillsToolboxAsync(options);
  },
};

export class SkillsToolboxBuilder extends SkillsFluent<SkillsToolboxBuilder> {
  build(): SkillsToolbox {
    return SkillsToolbox.of(this.toOptions());
  }

  buildTools(): ToolCallback[] {
    return this.build().tools as ToolCallback[];
  }

  buildAsync(): Promise<SkillsToolbox> {
    return SkillsToolbox.ofAsync(this.toOptions());
  }

  async buildToolsAsync(): Promise<ToolCallback[]> {
    return (await this.buildAsync()).tools as ToolCallback[];
  }
}

/**
 * Skill + file tools + optional mutation / HITL / web / memory / task tools.
 * Prefer {@link SkillsToolbox.builder}.
 */
export function skillsToolbox(options: SkillsToolboxOptions = {}): ToolCallback[] {
  return SkillsToolbox.of(options).tools as ToolCallback[];
}

export function createSkillsToolbox(options: SkillsToolboxOptions = {}): SkillsToolbox {
  const effectiveOptions = withAiIgnorePolicy(options);
  const discovery = semanticDiscoveryOptions(effectiveOptions);
  if (discovery.catalogStore) {
    throw new Error(
      'Asynchronous catalog stores require createSkillsToolboxAsync() or buildAsync()',
    );
  }
  const sourceResolution = resolveSkillSources(effectiveOptions);
  const files =
    effectiveOptions.files == null ? undefined : effectiveOptions.files.map(expandUserPath);
  const duplicates: SkillDuplicateDiagnostic[] = [];
  const suppressions: AiIgnoreSuppressionDiagnostic[] = [];
  const collected = collectSkills({
    ...effectiveOptions,
    directories: sourceResolution.directories,
    files,
    onDuplicate: (diagnostic) => {
      duplicates.push(diagnostic);
      effectiveOptions.onDuplicate?.(diagnostic);
    },
    onSuppressed: (diagnostic) => {
      suppressions.push(diagnostic);
      effectiveOptions.onSuppressed?.(diagnostic);
    },
  });

  return assembleSkillsToolbox(
    effectiveOptions,
    collected,
    undefined,
    sourceResolution,
    duplicates,
    suppressions,
  );
}

/** Build a toolbox without reading complete remote skill bodies during discovery. */
export async function createSkillsToolboxAsync(
  options: SkillsToolboxOptions = {},
): Promise<SkillsToolbox> {
  const discovery = semanticDiscoveryOptions(options);
  if (!discovery.catalogStore) return createSkillsToolbox(options);
  const effectiveOptions = withAiIgnorePolicy(options);
  const catalogStore = discovery.catalogStore;
  const health = await runSkillAdapterOperation(
    'Checking skill catalog health',
    () => catalogStore.health({ namespace: discovery.namespace }),
    discovery.timeoutMs,
  );
  if (health.status !== 'ready') {
    throw new SkillAdapterError('NOT_READY', health.message ?? 'Skill catalog is not ready');
  }
  if (discovery.vectorSearch) {
    const vectorSearch = discovery.vectorSearch;
    const vectorHealth = await runSkillAdapterOperation(
      'Checking skill vector index health',
      () => vectorSearch.health({ namespace: discovery.namespace }),
      discovery.timeoutMs,
    );
    if (vectorHealth.status !== 'ready') {
      throw new SkillAdapterError('NOT_READY', vectorHealth.message ?? 'Skill index is not ready');
    }
  }
  const descriptors = await runSkillAdapterOperation(
    'Listing skill catalog',
    () => catalogStore.list({ namespace: discovery.namespace }),
    discovery.timeoutMs,
  );
  if (descriptors.length === 0) throw new Error('At least one skill must be configured');
  for (const descriptor of descriptors) {
    const error =
      validateSkillName(descriptor.name) ?? validateSkillDescription(descriptor.description);
    if (error)
      throw new SkillAdapterError(
        'INVALID_RESPONSE',
        `Invalid descriptor '${descriptor.name}': ${error}`,
      );
  }
  return assembleSkillsToolbox(
    effectiveOptions,
    [],
    {
      store: catalogStore,
      descriptors,
      namespace: discovery.namespace,
    },
    { directories: [], sources: [], diagnostics: [] },
    [],
  );
}

function assembleSkillsToolbox(
  options: SkillsToolboxOptions,
  collected: readonly AgentSkill[],
  remote?: {
    readonly store: SkillCatalogStore;
    readonly descriptors: readonly SkillDescriptor[];
    readonly namespace?: string;
  },
  sourceResolution: ResolvedSkillSources = { directories: [], sources: [], diagnostics: [] },
  duplicateDiagnostics: readonly SkillDuplicateDiagnostic[] = [],
  suppressionDiagnostics: readonly AiIgnoreSuppressionDiagnostic[] = [],
): SkillsToolbox {
  for (const skill of collected) {
    validateSkill(skill, { matchDirectoryName: skill.basePath !== '.' });
  }

  const workspace = options.workspace ? expandUserPath(options.workspace) : process.cwd();
  const extra = (options.extraAllowedDirectories ?? []).map(expandUserPath);
  const skillDirectories = collected.map((skill) => skill.basePath).filter((path) => path !== '.');
  const allowedDirectories = uniqueResolvedRoots([workspace, ...skillDirectories, ...extra]);
  const runtime = createSkillsRuntime({
    workspace,
    extraAllowedDirectories: extra,
    skillDirectories,
    perSkillSandbox: options.perSkillSandbox,
  });
  const dirs = () => runtime.fileDirectories();
  const descriptors =
    remote?.descriptors ??
    collected.map((skill) => ({
      name: skill.name,
      description: skill.description,
      sourceHash: '',
    }));
  const skillsByName = new Map(collected.map((skill) => [skill.name, skill]));
  const aiIgnore: AiIgnoreToolPolicy | undefined =
    options.aiIgnore == null
      ? undefined
      : {
          enforcement: options.aiIgnore,
          policy: loadAiIgnorePolicy({ workspace }),
        };

  const buildSkillTool = (selected: readonly SkillDescriptor[]): ToolCallback =>
    gateToolCallback(
      remote
        ? asyncSkillsTool({
            ...options,
            descriptors: selected,
            catalogStore: remote.store,
            namespace: remote.namespace,
            timeoutMs: semanticDiscoveryOptions(options).timeoutMs,
            onActivate: (skill) => {
              runtime.activate(skill);
              options.onActivate?.(skill);
            },
          })
        : skillsTool({
            ...options,
            skills: selected
              .map((descriptor) => skillsByName.get(descriptor.name))
              .filter((skill): skill is AgentSkill => skill != null),
            directories: undefined,
            files: undefined,
            onActivate: (skill) => {
              runtime.activate(skill);
              options.onActivate?.(skill);
            },
          }),
      runtime,
    );

  const raw: ToolCallback[] = [
    buildSkillTool(descriptors),
    readTool({ allowedDirectories: dirs, aiIgnore }),
  ];

  if (options.list !== false) {
    raw.push(
      listDirectoryTool({ allowedDirectories: dirs, workingDirectory: workspace, aiIgnore }),
    );
  }
  if (options.glob !== false) {
    raw.push(
      globTool({
        allowedDirectories: dirs,
        workingDirectory: workspace,
        aiIgnorePolicy: options.aiIgnorePolicy,
        onSuppressed: options.onSuppressed,
      }),
    );
  }
  if (options.grep !== false) {
    raw.push(
      grepTool({
        allowedDirectories: dirs,
        workingDirectory: workspace,
        aiIgnorePolicy: options.aiIgnorePolicy,
        onSuppressed: options.onSuppressed,
      }),
    );
  }
  if (options.write) {
    raw.push(
      writeTool({ allowedDirectories: dirs, aiIgnore }),
      editTool({ allowedDirectories: dirs, aiIgnore }),
    );
  }
  if (options.shell) {
    raw.push(
      bashTool({
        allowedDirectories: dirs,
        workingDirectory: workspace,
        timeoutMs: options.shellTimeoutMs,
        confirm: options.confirmShell,
      }),
    );
  }
  if (options.todos !== false) {
    raw.push(todoWriteTool());
  }
  if (options.askUser) {
    raw.push(askUserQuestionTool({ questionHandler: options.askUser }));
  }

  const web = normalizeWeb(options.web);
  if (web.fetch) raw.push(webFetchTool());
  if (web.search) raw.push(webSearchTool({ apiKey: web.braveApiKey }));

  const memoriesDir = resolveMemoriesDir(options.memories, workspace);
  if (memoriesDir) {
    raw.push(...memoryTools({ directory: memoriesDir }));
  }

  const taskModel = resolveTaskModel(options);
  if (taskModel) {
    const withoutTask = raw.filter((tool) => tool.toolDefinition.name !== 'Task');
    raw.push(
      taskTool({
        chatModel: taskModel,
        tools: withoutTask,
        system: typeof options.task === 'object' ? options.task.system : undefined,
      }),
    );
  }

  const tools = raw.map((tool) =>
    tool.toolDefinition.name === (options.toolName ?? DEFAULT_SKILL_TOOL_NAME)
      ? tool
      : gateToolCallback(tool, runtime),
  );
  const retrievalAdvisor = createRetrievalAdvisor(
    options,
    workspace,
    collected,
    descriptors,
    buildSkillTool,
  );
  return {
    skills: collected,
    descriptors,
    allowedDirectories,
    tools,
    runtime,
    retrievalAdvisor,
    skillSources: sourceResolution.sources,
    skillDiagnostics: [
      ...sourceResolution.diagnostics,
      ...duplicateDiagnostics,
      ...suppressionDiagnostics,
    ],
  };
}

function withAiIgnorePolicy(options: SkillsToolboxOptions): SkillsToolboxOptions & {
  readonly aiIgnorePolicy: AiIgnorePolicy;
} {
  if (options.aiIgnorePolicy != null) return { ...options, aiIgnorePolicy: options.aiIgnorePolicy };
  const workspace = options.workspace ? expandUserPath(options.workspace) : process.cwd();
  return { ...options, aiIgnorePolicy: loadAiIgnorePolicy({ workspace }) };
}

function createRetrievalAdvisor(
  options: SkillsToolboxOptions,
  workspace: string,
  skills: readonly AgentSkill[],
  descriptors: readonly SkillDescriptor[],
  buildSkillTool: (descriptors: readonly SkillDescriptor[]) => ToolCallback,
): SkillsRetrievalAdvisor | undefined {
  if (options.semanticDiscovery === false) return undefined;
  const discovery =
    options.semanticDiscovery && typeof options.semanticDiscovery === 'object'
      ? options.semanticDiscovery
      : {};
  const configuredPath = discovery.indexFile ?? DEFAULT_SKILLS_INDEX_FILE;
  const indexFile = isAbsolute(configuredPath)
    ? resolve(expandUserPath(configuredPath))
    : resolve(workspace, configuredPath);
  const explicit =
    options.semanticDiscovery === true || typeof options.semanticDiscovery === 'object';
  if (!discovery.vectorSearch && !existsSync(indexFile)) {
    if (explicit) {
      throw new Error(
        `Skills index does not exist: ${indexFile}. Build it with SkillsIndex.builder()`,
      );
    }
    return undefined;
  }

  const index = discovery.vectorSearch ? undefined : loadSkillsIndex(indexFile);
  if (index && !index.metadata.indexed) return undefined;
  return new SkillsRetrievalAdvisor({
    index,
    skills,
    descriptors,
    catalogStore: discovery.catalogStore,
    vectorSearch: discovery.vectorSearch,
    namespace: discovery.namespace,
    timeoutMs: discovery.timeoutMs,
    embedder: discovery.embedder,
    limit: discovery.limit,
    minScore: discovery.minScore,
    recentUserMessages: discovery.recentUserMessages,
    toolName: options.toolName,
    toolOptions: {
      toolName: options.toolName,
      toolDescriptionTemplate: options.toolDescriptionTemplate,
      onActivate: options.onActivate,
    },
    // buildSkillTool already includes activation and the runtime gate.
    createTool: buildSkillTool,
  });
}

function semanticDiscoveryOptions(options: SkillsToolboxOptions): SkillsSemanticDiscoveryOptions {
  return options.semanticDiscovery && typeof options.semanticDiscovery === 'object'
    ? options.semanticDiscovery
    : {};
}

function normalizeWeb(web: SkillsToolboxOptions['web']): {
  fetch: boolean;
  search: boolean;
  braveApiKey?: string;
} {
  if (web === true) return { fetch: true, search: true };
  if (web == null || web === false) return { fetch: false, search: false };
  return {
    fetch: web.fetch !== false,
    search: web.search === true || web.braveApiKey != null,
    braveApiKey: web.braveApiKey,
  };
}

function resolveMemoriesDir(
  memories: SkillsToolboxOptions['memories'],
  workspace: string,
): string | undefined {
  if (memories === true) return `${workspace.replace(/[/\\]$/, '')}/.memory`;
  if (memories && typeof memories === 'object') return expandUserPath(memories.directory);
  return undefined;
}

function resolveTaskModel(options: SkillsToolboxOptions): ChatModel | undefined {
  if (options.task === false || options.task == null) return undefined;
  if (options.task === true) return options.chatModel;
  return options.task.chatModel;
}
