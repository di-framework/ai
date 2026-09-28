import {
  type CallAdvisor,
  type CallAdvisorChain,
  type ChatClientRequest,
  type ChatClientResponse,
  copyChatClientRequest,
  HIGHEST_PRECEDENCE,
  Prompt,
  type StreamAdvisor,
  type StreamAdvisorChain,
  type ToolCallback,
} from '@di-framework/ai';
import type { AgentSkill } from './parse-skill-markdown.ts';
import {
  runSkillAdapterOperation,
  SkillAdapterError,
  type SkillCatalogStore,
  type SkillChunkMatch,
  type SkillDescriptor,
  type SkillVectorSearch,
} from './skill-adapters.ts';
import type { SkillEmbedder } from './skill-embedder.ts';
import { TransformersJsSkillEmbedder } from './skill-embedder.ts';
import {
  assertSkillsIndexCurrent,
  LocalSkillVectorSearch,
  loadSkillsIndex,
  type SkillsIndex,
  type SkillsIndexMatch,
  searchSkillsIndex,
} from './skills-index.ts';
import {
  asyncSkillsTool,
  DEFAULT_SKILL_TOOL_NAME,
  type SkillsToolOptions,
  skillsTool,
} from './skills-tool.ts';

export const SKILLS_RETRIEVAL_CONTEXT = 'skills_retrieval';
export const DEFAULT_SKILLS_RETRIEVAL_ORDER = HIGHEST_PRECEDENCE + 250;

export interface SkillsRetrievalDiagnostic {
  readonly schema: '@di-framework/skills-retrieval-diagnostic';
  readonly version: 1;
  readonly decision: 'selected' | 'abstained' | 'error';
  readonly backend: 'local' | 'adapter';
  readonly matches: readonly SkillsIndexMatch[];
  readonly error?: string;
  readonly timings: {
    readonly loadMs: number;
    readonly embedMs: number;
    readonly searchMs: number;
    readonly totalMs: number;
  };
}

export interface SkillsRetrievalAdvisorOptions {
  readonly index?: SkillsIndex | string;
  readonly skills?: readonly AgentSkill[];
  readonly descriptors?: readonly SkillDescriptor[];
  readonly catalogStore?: SkillCatalogStore;
  readonly vectorSearch?: SkillVectorSearch;
  readonly namespace?: string;
  readonly timeoutMs?: number;
  readonly embedder?: SkillEmbedder;
  readonly limit?: number;
  readonly minScore?: number;
  readonly order?: number;
  readonly recentUserMessages?: number;
  readonly toolName?: string;
  readonly toolOptions?: Omit<SkillsToolOptions, 'skills' | 'directories' | 'files'>;
  /** Used by SkillsToolbox to preserve activation state and its runtime gate. */
  readonly createTool?: (descriptors: readonly SkillDescriptor[]) => ToolCallback;
  readonly onDiagnostic?: (diagnostic: SkillsRetrievalDiagnostic) => void;
}

/**
 * Replaces the full Skill catalog with the most relevant indexed subset before
 * the model sees the request. It runs after memory and before tool calling.
 */
export class SkillsRetrievalAdvisor implements CallAdvisor, StreamAdvisor {
  readonly name = 'Skills Retrieval Advisor';
  readonly order: number;

  private readonly index?: SkillsIndex;
  private readonly skillsByName: ReadonlyMap<string, AgentSkill>;
  private readonly descriptors: readonly SkillDescriptor[];
  private readonly catalogStore?: SkillCatalogStore;
  private readonly vectorSearch: SkillVectorSearch;
  private readonly namespace?: string;
  private readonly timeoutMs?: number;
  private readonly embedder?: SkillEmbedder;
  private readonly limit?: number;
  private readonly minScore?: number;
  private readonly recentUserMessages: number;
  private readonly toolName: string;
  private readonly toolOptions: Omit<SkillsToolOptions, 'skills' | 'directories' | 'files'>;
  private readonly createTool: (descriptors: readonly SkillDescriptor[]) => ToolCallback;
  private readonly onDiagnostic?: (diagnostic: SkillsRetrievalDiagnostic) => void;

  constructor(options: SkillsRetrievalAdvisorOptions) {
    this.index = typeof options.index === 'string' ? loadSkillsIndex(options.index) : options.index;
    if (!this.index && !options.vectorSearch) {
      throw new Error('Skills retrieval requires an index or vectorSearch adapter');
    }
    const skills = options.skills ?? [];
    if (this.index && skills.length > 0) {
      assertSkillsIndexCurrent(this.index, skills, { allowExtraSkills: true });
    }
    this.skillsByName = new Map(skills.map((skill) => [skill.name, skill]));
    this.descriptors =
      options.descriptors ??
      skills.map((skill) => ({
        name: skill.name,
        description: skill.description,
        sourceHash: '',
      }));
    this.catalogStore = options.catalogStore;
    this.vectorSearch =
      options.vectorSearch ?? new LocalSkillVectorSearch(this.index as SkillsIndex);
    this.namespace = options.namespace;
    this.timeoutMs = options.timeoutMs;
    this.embedder = options.embedder;
    this.limit = options.limit;
    this.minScore = options.minScore;
    this.order = options.order ?? DEFAULT_SKILLS_RETRIEVAL_ORDER;
    this.recentUserMessages = positiveInteger(
      options.recentUserMessages ?? 3,
      'recentUserMessages',
    );
    this.toolName = options.toolName ?? DEFAULT_SKILL_TOOL_NAME;
    this.toolOptions = options.toolOptions ?? {};
    this.onDiagnostic = options.onDiagnostic;
    this.createTool =
      options.createTool ??
      ((descriptors) => {
        if (this.catalogStore) {
          return asyncSkillsTool({
            ...this.toolOptions,
            descriptors,
            catalogStore: this.catalogStore,
            namespace: this.namespace,
            timeoutMs: this.timeoutMs,
            toolName: this.toolName,
          });
        }
        return skillsTool({
          ...this.toolOptions,
          skills: descriptors
            .map((descriptor) => this.skillsByName.get(descriptor.name))
            .filter((skill): skill is AgentSkill => skill != null),
          toolName: this.toolName,
        });
      });
  }

  async before(request: ChatClientRequest): Promise<ChatClientRequest> {
    if (this.index && !this.index.metadata.indexed) return request;
    const callbacks = request.prompt.options?.toolCallbacks;
    if (!callbacks?.some((tool) => tool.toolDefinition.name === this.toolName)) return request;

    const task = this.taskText(request);
    const started = performance.now();
    const backend =
      this.index && !this.catalogStore && this.vectorSearch instanceof LocalSkillVectorSearch
        ? 'local'
        : 'adapter';
    let semantic: readonly SkillsIndexMatch[];
    try {
      semantic =
        backend === 'local'
          ? await searchSkillsIndex(this.index as SkillsIndex, task, {
              embedder: this.embedder,
              limit: this.limit,
              minScore: this.minScore,
            })
          : await this.searchAdapter(task);
    } catch (error) {
      const totalMs = performance.now() - started;
      this.emitDiagnostic('error', backend, [], totalMs, error);
      throw error;
    }
    const matches = this.pinExplicitSkillNames(task, semantic);
    const descriptorsByName = new Map(
      this.descriptors.map((descriptor) => [descriptor.name, descriptor]),
    );
    const selected = matches
      .map((match) => descriptorsByName.get(match.name))
      .filter((descriptor): descriptor is SkillDescriptor => descriptor != null);

    if (selected.length === 0) {
      this.emitDiagnostic('abstained', backend, [], performance.now() - started);
      request.context.set(SKILLS_RETRIEVAL_CONTEXT, { decision: 'abstained', matches: [] });
      return copyChatClientRequest(request, {
        prompt: new Prompt(request.prompt.messages, {
          ...request.prompt.options,
          toolCallbacks: callbacks.filter((tool) => tool.toolDefinition.name !== this.toolName),
        }),
        context: request.context,
      });
    }

    const replacement = this.createTool(selected);
    const toolCallbacks = callbacks.map((tool) =>
      tool.toolDefinition.name === this.toolName ? replacement : tool,
    );
    request.context.set(SKILLS_RETRIEVAL_CONTEXT, matches);
    this.emitDiagnostic('selected', backend, matches, performance.now() - started);
    return copyChatClientRequest(request, {
      prompt: new Prompt(request.prompt.messages, {
        ...request.prompt.options,
        toolCallbacks,
      }),
      context: request.context,
    });
  }

  private emitDiagnostic(
    decision: SkillsRetrievalDiagnostic['decision'],
    backend: SkillsRetrievalDiagnostic['backend'],
    matches: readonly SkillsIndexMatch[],
    totalMs: number,
    error?: unknown,
  ): void {
    this.onDiagnostic?.({
      schema: '@di-framework/skills-retrieval-diagnostic',
      version: 1,
      decision,
      backend,
      matches,
      error: error == null ? undefined : error instanceof Error ? error.message : String(error),
      timings: { loadMs: 0, embedMs: 0, searchMs: totalMs, totalMs },
    });
  }

  async adviseCall(
    request: ChatClientRequest,
    chain: CallAdvisorChain,
  ): Promise<ChatClientResponse> {
    return chain.nextCall(await this.before(request));
  }

  async *adviseStream(
    request: ChatClientRequest,
    chain: StreamAdvisorChain,
  ): AsyncIterable<ChatClientResponse> {
    yield* chain.nextStream(await this.before(request));
  }

  private taskText(request: ChatClientRequest): string {
    const messages = request.prompt.getUserMessages().slice(-this.recentUserMessages);
    const task = messages
      .map((message) => message.text ?? '')
      .filter((text) => text.trim().length > 0)
      .join('\n\n');
    return task || request.prompt.getContents();
  }

  private pinExplicitSkillNames(
    task: string,
    semantic: readonly SkillsIndexMatch[],
  ): readonly SkillsIndexMatch[] {
    const limit = this.limit ?? this.index?.metadata.retrievalLimit ?? 10;
    const lowerTask = task.toLowerCase();
    const explicit = this.descriptors
      .filter((descriptor) => containsSkillName(lowerTask, descriptor.name))
      .map((descriptor) => ({
        name: descriptor.name,
        description: descriptor.description ?? '',
        score: 1,
        matchedChunk: 0,
        matchedSource: 'document' as const,
      }));
    if (explicit.length === 0) return semantic;

    const merged = new Map<string, SkillsIndexMatch>();
    for (const match of [...explicit, ...semantic]) {
      if (!merged.has(match.name)) merged.set(match.name, match);
    }
    return [...merged.values()].slice(0, Math.max(limit, explicit.length));
  }

  private async searchAdapter(task: string): Promise<readonly SkillsIndexMatch[]> {
    const metadata = await runSkillAdapterOperation(
      'Loading skill vector metadata',
      () => this.vectorSearch.metadata({ namespace: this.namespace }),
      this.timeoutMs,
    );
    const catalogStore = this.catalogStore;
    const catalogVersion = catalogStore
      ? await runSkillAdapterOperation(
          'Loading skill catalog version',
          () => catalogStore.version({ namespace: this.namespace }),
          this.timeoutMs,
        )
      : this.index?.metadata.catalogHash;
    if (catalogVersion == null) {
      throw new SkillAdapterError('STALE_CATALOG', 'Catalog version is unavailable');
    }
    const embedder =
      this.embedder ??
      new TransformersJsSkillEmbedder({ model: metadata.model, revision: metadata.revision });
    const [query] = await runSkillAdapterOperation(
      'Embedding skill query',
      () => embedder.embed([task], { purpose: 'query' }),
      this.timeoutMs,
    );
    if (!query) throw new SkillAdapterError('INVALID_RESPONSE', 'Embedder omitted query vector');
    const limit = this.limit ?? this.index?.metadata.retrievalLimit ?? 10;
    const chunks = await runSkillAdapterOperation(
      'Searching skill vectors',
      () =>
        this.vectorSearch.query(query, {
          namespace: this.namespace,
          catalogVersion,
          model: embedder.model,
          revision: embedder.revision,
          embedderId: embedder.id,
          limit: Math.max(limit * 8, limit),
          minScore: this.minScore,
        }),
      this.timeoutMs,
    );
    return aggregateSkillChunkMatches(
      chunks,
      limit,
      new Set(this.descriptors.map(({ name }) => name)),
    );
  }
}

/** Shared skill-level MaxSim aggregation for local and hosted chunk search. */
export function aggregateSkillChunkMatches(
  chunks: readonly SkillChunkMatch[],
  limit: number,
  knownSkills?: ReadonlySet<string>,
): readonly SkillsIndexMatch[] {
  const grouped = new Map<string, { description: string; best: SkillChunkMatch; first?: number }>();
  for (const chunk of chunks) {
    if (!Number.isFinite(chunk.score) || chunk.chunk < 0 || !Number.isInteger(chunk.chunk)) {
      throw new SkillAdapterError(
        'PARTIAL_RESULT',
        'Vector search returned an invalid chunk match',
      );
    }
    if (knownSkills && !knownSkills.has(chunk.name)) {
      throw new SkillAdapterError(
        'PARTIAL_RESULT',
        `Vector search returned unknown skill '${chunk.name}'`,
      );
    }
    const current = grouped.get(chunk.name);
    if (!current) {
      grouped.set(chunk.name, {
        description: chunk.description,
        best: chunk,
        first: chunk.chunk === 0 ? chunk.score : undefined,
      });
    } else {
      if (chunk.score > current.best.score) current.best = chunk;
      if (chunk.chunk === 0) current.first = chunk.score;
    }
  }
  return [...grouped.entries()]
    .map(([name, group]) => ({
      name,
      description: group.description,
      score: group.first == null ? group.best.score : 0.75 * group.first + 0.25 * group.best.score,
      matchedChunk: group.best.chunk,
      matchedSource: group.best.source,
    }))
    .sort((left, right) => right.score - left.score || left.name.localeCompare(right.name))
    .slice(0, limit);
}

function containsSkillName(lowerTask: string, name: string): boolean {
  const lowerName = name.toLowerCase();
  let start = lowerTask.indexOf(lowerName);
  while (start >= 0) {
    const before = start === 0 ? '' : lowerTask.charAt(start - 1);
    const afterIndex = start + lowerName.length;
    const after = afterIndex === lowerTask.length ? '' : lowerTask.charAt(afterIndex);
    if (!/[a-z0-9-]/.test(before) && !/[a-z0-9-]/.test(after)) return true;
    start = lowerTask.indexOf(lowerName, start + 1);
  }
  return false;
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) throw new Error(`${name} must be a positive integer`);
  return value;
}
