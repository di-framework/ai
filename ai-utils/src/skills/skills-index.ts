import { Buffer } from 'node:buffer';
import { createHash, randomUUID } from 'node:crypto';
import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { expandUserPath } from '../sandbox/paths.ts';
import type { AgentSkill } from './parse-skill-markdown.ts';
import {
  assertReadyMetadata,
  type SkillAdapterCapabilities,
  type SkillAdapterHealth,
  type SkillChunkMatch,
  type SkillIndexWriteReceipt,
  type SkillIndexWriteRequest,
  type SkillIndexWriter,
  type SkillVectorIndexMetadata,
  type SkillVectorQueryOptions,
  type SkillVectorSearch,
} from './skill-adapters.ts';
import { type SkillEmbedder, TransformersJsSkillEmbedder } from './skill-embedder.ts';
import { asSkillsIndexOperationError, SkillsIndexOperationError } from './skills-index-errors.ts';
import { collectSkills } from './skills-tool.ts';
import { validateSkill } from './validate-skill.ts';

export const SKILLS_INDEX_FORMAT = '@di-framework/ai-utils/skills-index';
export const SKILLS_INDEX_VERSION = 3;
export const SKILLS_INDEX_VECTOR_ENCODING = 'int8-per-vector-v1';
export const SKILLS_INDEX_V2_VECTOR_ENCODING = 'float32-le-base64';
export const SKILLS_INDEX_SCORING = 'hybrid-rrf-bm25-v1';
export const SKILLS_INDEX_V2_SCORING = 'frontmatter-guided-document-cosine-v1';
export const SKILLS_INDEX_FIRST_CHUNK_WEIGHT = 0.75;
export const DEFAULT_SKILLS_INDEX_FILE = '.di-framework/skills-index.json';
export const DEFAULT_SKILLS_INDEX_THRESHOLD = 50;
export const DEFAULT_SKILLS_RETRIEVAL_LIMIT = 10;
export const DEFAULT_SKILLS_INDEX_BATCH_SIZE = 32;
export const DEFAULT_SKILLS_INDEX_CHUNK_TOKENS = 256;
export const DEFAULT_SKILLS_INDEX_CHUNK_OVERLAP_TOKENS = 32;
export const DEFAULT_SKILLS_INDEX_SCORING_PARAMETERS: SkillsIndexScoringParameters = {
  denseWeight: 1,
  lexicalWeight: 1,
  rrfK: 60,
  bm25K1: 1.2,
  bm25B: 0.75,
  abstentionThreshold: 0.018,
};

export type SkillsIndexChunkSource = 'document';

export interface SkillsIndexMetadata {
  readonly kind: typeof SKILLS_INDEX_FORMAT;
  readonly version: 2 | typeof SKILLS_INDEX_VERSION;
  readonly indexed: boolean;
  readonly skillCount: number;
  readonly chunkCount: number;
  readonly threshold: number;
  readonly retrievalLimit: number;
  readonly chunkTokens: number;
  readonly chunkOverlapTokens: number;
  readonly scoring: typeof SKILLS_INDEX_SCORING | typeof SKILLS_INDEX_V2_SCORING;
  readonly vectorEncoding:
    | typeof SKILLS_INDEX_VECTOR_ENCODING
    | typeof SKILLS_INDEX_V2_VECTOR_ENCODING;
  readonly catalogHash: string;
  readonly model?: string;
  readonly revision?: string;
  readonly embedderId?: string;
  readonly dimensions?: number;
  readonly vectorFile?: string;
  readonly vectorHash?: string;
  readonly vectorBytes?: number;
  readonly scoringParameters?: SkillsIndexScoringParameters;
}

export interface SkillsIndexScoringParameters {
  readonly denseWeight: number;
  readonly lexicalWeight: number;
  readonly rrfK: number;
  readonly bm25K1: number;
  readonly bm25B: number;
  readonly abstentionThreshold: number;
}

export interface SkillsLexicalIndex {
  readonly documentLengths: readonly number[];
  readonly averageDocumentLength: number;
  /** token -> compact [document index, weighted term frequency] pairs */
  readonly postings: Readonly<Record<string, readonly number[]>>;
}

export interface QuantizedSkillVector {
  readonly values: Int8Array;
  readonly scale: number;
  readonly norm: number;
}

export interface SkillsIndexChunk {
  readonly source: SkillsIndexChunkSource;
  readonly embedding: Float32Array | QuantizedSkillVector;
  /** Build-time source chunk; intentionally omitted from generated artifacts. */
  readonly text?: string;
}

export interface SkillsIndexEntry {
  readonly kind: 'skill';
  readonly name: string;
  readonly description: string;
  readonly documentHash: string;
  readonly chunks: readonly SkillsIndexChunk[];
}

export interface SkillsIndex {
  readonly file?: string;
  readonly metadata: SkillsIndexMetadata;
  readonly entries: readonly SkillsIndexEntry[];
  readonly lexical?: SkillsLexicalIndex;
}

export interface BuildSkillsIndexOptions {
  readonly skills?: readonly AgentSkill[];
  readonly directories?: readonly string[];
  readonly files?: readonly string[];
  readonly outputFile?: string;
  /** Build embeddings only when skill count is strictly greater than this value. */
  readonly threshold?: number;
  readonly retrievalLimit?: number;
  readonly batchSize?: number;
  readonly chunkTokens?: number;
  readonly chunkOverlapTokens?: number;
  readonly embedder?: SkillEmbedder;
  /** Optional hosted/object-storage writer. Local JSONL remains the default. */
  readonly writer?: SkillIndexWriter;
  readonly force?: boolean;
  /** Reports embedded chunks, not skills. */
  readonly onProgress?: (completed: number, total: number) => void;
}

export interface BuildSkillsIndexResult {
  readonly outputFile: string;
  readonly indexed: boolean;
  readonly skillCount: number;
  readonly chunkCount: number;
  readonly dimensions?: number;
  readonly unchanged?: boolean;
  readonly receipt?: SkillIndexWriteReceipt;
}

/** Preferred build-time factory for a semantic Agent Skills index. */
export const SkillsIndex = {
  builder(): SkillsIndexBuilder {
    return new SkillsIndexBuilder();
  },
};

/** Fluent builder for the generated JSONL skills index. */
export class SkillsIndexBuilder {
  private readonly draft: {
    -readonly [K in keyof BuildSkillsIndexOptions]?: BuildSkillsIndexOptions[K];
  } = {};

  addSkill(skill: AgentSkill): this {
    this.draft.skills = [...(this.draft.skills ?? []), skill];
    return this;
  }

  addSkills(skills: readonly AgentSkill[]): this {
    this.draft.skills = [...(this.draft.skills ?? []), ...skills];
    return this;
  }

  addSkillsDirectory(directory: string): this {
    this.draft.directories = [...(this.draft.directories ?? []), directory];
    return this;
  }

  addSkillsDirectories(directories: readonly string[]): this {
    this.draft.directories = [...(this.draft.directories ?? []), ...directories];
    return this;
  }

  addSkillsFile(skillMdPath: string): this {
    this.draft.files = [...(this.draft.files ?? []), skillMdPath];
    return this;
  }

  addSkillsFiles(skillMdPaths: readonly string[]): this {
    this.draft.files = [...(this.draft.files ?? []), ...skillMdPaths];
    return this;
  }

  outputFile(path: string): this {
    this.draft.outputFile = path;
    return this;
  }

  threshold(count: number): this {
    this.draft.threshold = count;
    return this;
  }

  retrievalLimit(count: number): this {
    this.draft.retrievalLimit = count;
    return this;
  }

  batchSize(count: number): this {
    this.draft.batchSize = count;
    return this;
  }

  chunkTokens(count: number): this {
    this.draft.chunkTokens = count;
    return this;
  }

  chunkOverlapTokens(count: number): this {
    this.draft.chunkOverlapTokens = count;
    return this;
  }

  embedder(embedder: SkillEmbedder): this {
    this.draft.embedder = embedder;
    return this;
  }

  writer(writer: SkillIndexWriter): this {
    this.draft.writer = writer;
    return this;
  }

  force(enabled = true): this {
    this.draft.force = enabled;
    return this;
  }

  onProgress(handler: (completed: number, total: number) => void): this {
    this.draft.onProgress = handler;
    return this;
  }

  toOptions(): BuildSkillsIndexOptions {
    return { ...this.draft };
  }

  build(): Promise<BuildSkillsIndexResult> {
    return buildSkillsIndex(this.toOptions());
  }
}

export interface SearchSkillsIndexOptions {
  readonly embedder?: SkillEmbedder;
  readonly limit?: number;
  readonly minScore?: number;
  readonly abstentionThreshold?: number;
}

export interface SkillsIndexMatch {
  readonly name: string;
  readonly description: string;
  readonly score: number;
  readonly matchedChunk: number;
  readonly matchedSource: SkillsIndexChunkSource;
  readonly denseScore?: number;
  readonly lexicalScore?: number;
  readonly exactName?: boolean;
}

export interface SkillsIndexEntryScore {
  readonly score: number;
  readonly matchedChunk: number;
  readonly matchedSource: SkillsIndexChunkSource;
}

/** Exact SKILL.md source used for chunking and catalog hashing. */
export function skillIndexText(skill: Pick<AgentSkill, 'source'>): string {
  return skill.source;
}

/**
 * Build a deterministic JSONL artifact. Catalogs at or below the threshold
 * write only metadata and never initialize Transformers.js.
 */
export async function buildSkillsIndex(
  options: BuildSkillsIndexOptions = {},
): Promise<BuildSkillsIndexResult> {
  try {
    return await buildSkillsIndexUnchecked(options);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const code = /must be|must not|At least one|Duplicate skill/.test(message)
      ? 'INVALID_OPTIONS'
      : /does not exist|not a directory|not a file/.test(message)
        ? 'SOURCE_NOT_FOUND'
        : /embed|embedding|tokenizer|chunk vector|dimensions/.test(message)
          ? 'EMBEDDING_FAILED'
          : 'OPERATION_FAILED';
    throw asSkillsIndexOperationError('build', code, error);
  }
}

async function buildSkillsIndexUnchecked(
  options: BuildSkillsIndexOptions,
): Promise<BuildSkillsIndexResult> {
  const threshold = nonNegativeInteger(
    options.threshold ?? DEFAULT_SKILLS_INDEX_THRESHOLD,
    'threshold',
  );
  const retrievalLimit = positiveInteger(
    options.retrievalLimit ?? DEFAULT_SKILLS_RETRIEVAL_LIMIT,
    'retrievalLimit',
  );
  const batchSize = positiveInteger(
    options.batchSize ?? DEFAULT_SKILLS_INDEX_BATCH_SIZE,
    'batchSize',
  );
  const chunkTokens = positiveInteger(
    options.chunkTokens ?? DEFAULT_SKILLS_INDEX_CHUNK_TOKENS,
    'chunkTokens',
  );
  const chunkOverlapTokens = nonNegativeInteger(
    options.chunkOverlapTokens ?? DEFAULT_SKILLS_INDEX_CHUNK_OVERLAP_TOKENS,
    'chunkOverlapTokens',
  );
  if (chunkOverlapTokens >= chunkTokens) {
    throw new Error('chunkOverlapTokens must be smaller than chunkTokens');
  }

  const outputFile = resolve(expandUserPath(options.outputFile ?? DEFAULT_SKILLS_INDEX_FILE));
  const skills = collectSkills(options)
    .slice()
    .sort((left, right) => left.name.localeCompare(right.name));

  for (const skill of skills) {
    validateSkill(skill, { matchDirectoryName: skill.basePath !== '.' });
  }

  const catalogHash = hashSkillCatalog(skills);
  const commonMetadata = {
    kind: SKILLS_INDEX_FORMAT,
    version: SKILLS_INDEX_VERSION,
    skillCount: skills.length,
    threshold,
    retrievalLimit,
    chunkTokens,
    chunkOverlapTokens,
    scoring: SKILLS_INDEX_SCORING,
    scoringParameters: DEFAULT_SKILLS_INDEX_SCORING_PARAMETERS,
    vectorEncoding: SKILLS_INDEX_VECTOR_ENCODING,
    catalogHash,
  } as const;

  if (skills.length <= threshold) {
    const metadata: SkillsIndexMetadata = {
      ...commonMetadata,
      indexed: false,
      chunkCount: 0,
    };
    try {
      new LocalSkillIndexWriter(outputFile).writeIndex({ metadata, entries: [] });
    } catch (error) {
      throw new SkillsIndexOperationError('build', 'WRITE_FAILED', 'Failed to write skills index', {
        cause: error,
      });
    }
    return { outputFile, indexed: false, skillCount: skills.length, chunkCount: 0 };
  }

  const embedder = options.embedder ?? new TransformersJsSkillEmbedder();
  if (!options.force && existsSync(outputFile)) {
    try {
      const existing = loadSkillsIndex(outputFile);
      if (
        existing.metadata.indexed &&
        existing.metadata.catalogHash === catalogHash &&
        existing.metadata.threshold === threshold &&
        existing.metadata.retrievalLimit === retrievalLimit &&
        existing.metadata.chunkTokens === chunkTokens &&
        existing.metadata.chunkOverlapTokens === chunkOverlapTokens &&
        existing.metadata.embedderId === embedder.id
      ) {
        assertSkillsIndexCurrent(existing, skills);
        return {
          outputFile,
          indexed: true,
          skillCount: skills.length,
          chunkCount: existing.metadata.chunkCount,
          dimensions: existing.metadata.dimensions,
          unchanged: true,
        };
      }
    } catch {
      // A malformed or stale generated artifact is replaced below.
    }
  }

  const pendingChunks: Array<{
    skillIndex: number;
    source: SkillsIndexChunkSource;
    text: string;
  }> = [];
  for (let skillIndex = 0; skillIndex < skills.length; skillIndex++) {
    const skill = skills[skillIndex];
    if (!skill) continue;
    const documentChunks = await embedder.split(skillIndexText(skill), {
      maxTokens: chunkTokens,
      overlapTokens: chunkOverlapTokens,
    });
    for (const text of documentChunks) {
      pendingChunks.push({ skillIndex, source: 'document', text });
    }
  }
  if (pendingChunks.length === 0) throw new Error('Skill tokenizer returned no chunks');

  const chunksBySkill: SkillsIndexChunk[][] = skills.map(() => []);
  let dimensions: number | undefined;
  for (let start = 0; start < pendingChunks.length; start += batchSize) {
    const batch = pendingChunks.slice(start, start + batchSize);
    const embedded = await embedder.embed(
      batch.map((chunk) => chunk.text),
      { purpose: 'document' },
    );
    if (embedded.length !== batch.length) {
      throw new Error(
        `Skill embedder returned ${embedded.length} vectors for a batch of ${batch.length}`,
      );
    }
    for (let offset = 0; offset < embedded.length; offset++) {
      const chunk = batch[offset];
      const input = embedded[offset];
      if (!chunk || !input) throw new Error('Skill embedder omitted a chunk vector');
      const vector = normalizeVector(input);
      dimensions ??= vector.length;
      if (vector.length !== dimensions) {
        throw new Error(`Skill embedding has ${vector.length} dimensions; expected ${dimensions}`);
      }
      chunksBySkill[chunk.skillIndex]?.push({
        source: chunk.source,
        embedding: vector,
        text: chunk.text,
      });
    }
    options.onProgress?.(
      Math.min(start + batch.length, pendingChunks.length),
      pendingChunks.length,
    );
  }

  if (dimensions == null || dimensions === 0) {
    throw new Error('Skill embedder returned empty vectors');
  }

  const metadata: SkillsIndexMetadata = {
    ...commonMetadata,
    indexed: true,
    chunkCount: pendingChunks.length,
    model: embedder.model,
    revision: embedder.revision,
    embedderId: embedder.id,
    dimensions,
  };
  const entries: SkillsIndexEntry[] = skills.map((skill, index) => {
    const chunks = chunksBySkill[index];
    if (!chunks || chunks.length === 0) {
      throw new Error(`Missing chunks for skill '${skill.name}'`);
    }
    return {
      kind: 'skill',
      name: skill.name,
      description: skill.description ?? '',
      documentHash: hashText(skillIndexText(skill)),
      chunks,
    };
  });
  const index = { metadata, entries, lexical: buildLexicalIndex(skills) } satisfies SkillsIndex;
  let receipt: SkillIndexWriteReceipt | undefined;
  try {
    if (options.writer) {
      receipt = await options.writer.replace(toSkillIndexWriteRequest(index));
      if (!receipt.ready) throw new Error('Skill index writer did not return a ready receipt');
    } else {
      new LocalSkillIndexWriter(outputFile).writeIndex(index);
    }
  } catch (error) {
    throw new SkillsIndexOperationError('build', 'WRITE_FAILED', 'Failed to write skills index', {
      cause: error,
    });
  }
  return {
    outputFile,
    indexed: true,
    skillCount: skills.length,
    chunkCount: pendingChunks.length,
    dimensions,
    receipt,
  };
}

const LOCAL_SEARCH_CAPABILITIES: SkillAdapterCapabilities = {
  namespaces: false,
  lazyBodies: false,
  vectorSearch: true,
  indexWriting: false,
  eventuallyConsistent: false,
};

/** Filesystem JSONL plus exact cosine search behind the platform contract. */
export class LocalSkillVectorSearch implements SkillVectorSearch {
  readonly capabilities = LOCAL_SEARCH_CAPABILITIES;
  readonly index: SkillsIndex;

  constructor(index: SkillsIndex | string = DEFAULT_SKILLS_INDEX_FILE) {
    this.index = typeof index === 'string' ? loadSkillsIndex(index) : index;
  }

  metadata(): Promise<SkillVectorIndexMetadata> {
    return Promise.resolve(adapterMetadata(this.index));
  }

  async health(): Promise<SkillAdapterHealth> {
    const metadata = adapterMetadata(this.index);
    return metadata.ready
      ? { status: 'ready', checkedVersion: metadata.indexVersion }
      : { status: 'not-ready', message: 'The local index is below its indexing threshold' };
  }

  async query(
    vector: ArrayLike<number>,
    options: SkillVectorQueryOptions = {},
  ): Promise<readonly SkillChunkMatch[]> {
    const metadata = adapterMetadata(this.index);
    assertReadyMetadata(metadata, options);
    if (options.namespace != null)
      throw new Error('Local JSONL search does not support namespaces');
    if (vector.length !== metadata.dimensions) {
      throw new Error(
        `Query embedding has ${vector.length} dimensions; expected ${metadata.dimensions}`,
      );
    }
    const limit = positiveInteger(options.limit ?? this.index.metadata.retrievalLimit, 'limit');
    const minScore = options.minScore ?? Number.NEGATIVE_INFINITY;
    if (Number.isNaN(minScore)) throw new Error('minScore must be a number');
    return this.index.entries
      .flatMap((entry) =>
        entry.chunks.map((chunk, chunkIndex) => ({
          name: entry.name,
          description: entry.description,
          score: skillVectorSimilarity(vector, chunk.embedding),
          chunk: chunkIndex,
          source: chunk.source,
        })),
      )
      .filter((match) => match.score >= minScore)
      .sort(
        (left, right) =>
          right.score - left.score ||
          left.name.localeCompare(right.name) ||
          left.chunk - right.chunk,
      )
      .slice(0, limit);
  }
}

const LOCAL_WRITER_CAPABILITIES: SkillAdapterCapabilities = {
  ...LOCAL_SEARCH_CAPABILITIES,
  vectorSearch: false,
  indexWriting: true,
};

/** Atomic local JSONL writer used by the compatible synchronous builder path. */
export class LocalSkillIndexWriter implements SkillIndexWriter {
  readonly capabilities = LOCAL_WRITER_CAPABILITIES;

  constructor(readonly file = DEFAULT_SKILLS_INDEX_FILE) {}

  replace(request: SkillIndexWriteRequest): Promise<SkillIndexWriteReceipt> {
    const entries = groupWriteVectors(request);
    const metadata: SkillsIndexMetadata = {
      kind: SKILLS_INDEX_FORMAT,
      version: SKILLS_INDEX_VERSION,
      indexed: true,
      skillCount: entries.length,
      chunkCount: request.vectors.length,
      threshold: 0,
      retrievalLimit: Math.max(1, entries.length),
      chunkTokens: DEFAULT_SKILLS_INDEX_CHUNK_TOKENS,
      chunkOverlapTokens: DEFAULT_SKILLS_INDEX_CHUNK_OVERLAP_TOKENS,
      scoring: SKILLS_INDEX_SCORING,
      scoringParameters: DEFAULT_SKILLS_INDEX_SCORING_PARAMETERS,
      vectorEncoding: SKILLS_INDEX_VECTOR_ENCODING,
      catalogHash: request.metadata.catalogVersion,
      model: request.metadata.model,
      revision: request.metadata.revision,
      embedderId: request.metadata.embedderId,
      dimensions: request.metadata.dimensions,
    };
    this.writeIndex({ metadata, entries });
    return Promise.resolve({
      ...request.metadata,
      ready: true,
      writtenVectors: request.vectors.length,
    });
  }

  writeIndex(index: SkillsIndex): void {
    writeV3IndexAtomically(resolve(expandUserPath(this.file)), index);
  }
}

/** Parse and validate a generated skill index. */
export function loadSkillsIndex(file = DEFAULT_SKILLS_INDEX_FILE): SkillsIndex {
  const absolute = resolve(expandUserPath(file));
  let text: string;
  try {
    text = readFileSync(absolute, 'utf8');
  } catch (error) {
    throw new Error(`Skills index does not exist or cannot be read: ${absolute}`, { cause: error });
  }
  if (text.trimStart().startsWith('{')) {
    let manifest:
      | { metadata?: Partial<SkillsIndexMetadata>; entries?: unknown; lexical?: unknown }
      | undefined;
    try {
      manifest = JSON.parse(text);
    } catch {
      // Version-2 JSONL also begins with an object and is parsed below.
    }
    if (manifest?.metadata?.version === SKILLS_INDEX_VERSION) {
      return parseV3Index(manifest, absolute);
    }
    if (manifest?.metadata?.kind === SKILLS_INDEX_FORMAT) throw unsupportedIndexError(absolute);
  }
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length === 0) throw new Error(`Skills index is empty: ${absolute}`);

  const metadataLine = lines[0];
  if (metadataLine == null) throw new Error(`Skills index is empty: ${absolute}`);
  const metadata = parseMetadata(metadataLine, absolute, 2);
  const entries = lines
    .slice(1)
    .map((line, index) => parseEntry(line, index + 2, absolute, metadata.dimensions));
  if (!metadata.indexed && entries.length > 0) {
    throw new Error(`Non-indexed skills artifact contains vector entries: ${absolute}`);
  }
  if (metadata.indexed && entries.length !== metadata.skillCount) {
    throw new Error(
      `Skills index contains ${entries.length} entries but metadata declares ${metadata.skillCount}: ${absolute}`,
    );
  }
  if (metadata.indexed) {
    const names = new Set<string>();
    let chunkCount = 0;
    for (const entry of entries) {
      if (names.has(entry.name)) throw new Error(`Duplicate skill '${entry.name}' in ${absolute}`);
      names.add(entry.name);
      chunkCount += entry.chunks.length;
    }
    if (chunkCount !== metadata.chunkCount) {
      throw new Error(
        `Skills index contains ${chunkCount} chunks but metadata declares ${metadata.chunkCount}: ${absolute}`,
      );
    }
  }
  return { file: absolute, metadata, entries };
}

/** Fail when indexed metadata or SKILL.md source no longer matches loaded skills. */
export function assertSkillsIndexCurrent(
  index: SkillsIndex,
  skills: readonly AgentSkill[],
  options: { allowExtraSkills?: boolean } = {},
): void {
  const map = new Map(skills.map((skill) => [skill.name, skill]));
  for (const entry of index.entries) {
    const skill = map.get(entry.name);
    if (!skill) {
      throw staleIndexError(index, `indexed skill '${entry.name}' is missing`);
    }
    if ((skill.description ?? '') !== entry.description) {
      throw staleIndexError(index, `description changed for '${entry.name}'`);
    }
    if (hashText(skillIndexText(skill)) !== entry.documentHash) {
      throw staleIndexError(index, `SKILL.md changed for '${entry.name}'`);
    }
  }
  if (!options.allowExtraSkills && hashSkillCatalog(skills) !== index.metadata.catalogHash) {
    throw staleIndexError(index, 'the loaded skill catalog changed');
  }
}

/** Embed a task and rank skills with frontmatter-guided SKILL.md chunk scores. */
export async function searchSkillsIndex(
  index: SkillsIndex,
  task: string,
  options: SearchSkillsIndexOptions = {},
): Promise<readonly SkillsIndexMatch[]> {
  if (!index.metadata.indexed) return [];
  const embedder =
    options.embedder ??
    new TransformersJsSkillEmbedder({
      model: index.metadata.model,
      revision: index.metadata.revision,
    });
  assertCompatibleEmbedder(index, embedder);
  const [query] = await embedder.embed([task], { purpose: 'query' });
  if (!query) throw new Error('Skill embedder did not return a query vector');
  return rankHybridSkillsIndex(index, query, task, options);
}

/** Deterministic dense + BM25 reciprocal-rank fusion with exact-name pinning. */
export function rankHybridSkillsIndex(
  index: SkillsIndex,
  query: ArrayLike<number>,
  task: string,
  options: Pick<SearchSkillsIndexOptions, 'limit' | 'minScore' | 'abstentionThreshold'> = {},
): readonly SkillsIndexMatch[] {
  const limit = positiveInteger(options.limit ?? index.metadata.retrievalLimit, 'limit');
  const dense = rankSkillsIndex(index, query, {
    limit: index.entries.length,
    minScore: options.minScore,
  });
  if (!index.lexical || index.metadata.version === 2) return dense.slice(0, limit);
  const parameters = index.metadata.scoringParameters ?? DEFAULT_SKILLS_INDEX_SCORING_PARAMETERS;
  const lexicalByIndex = scoreLexicalIndex(index.lexical, task, parameters);
  const lexicalScores = new Map(
    [...lexicalByIndex].map(([entryIndex, score]) => [
      index.entries[entryIndex]?.name ?? '',
      score,
    ]),
  );
  const lexical = [...lexicalScores.entries()].sort(
    ([leftName, left], [rightName, right]) => right - left || leftName.localeCompare(rightName),
  );
  const denseRanks = new Map(dense.map((match, rank) => [match.name, rank + 1]));
  const lexicalRanks = new Map(lexical.map(([name], rank) => [name, rank + 1]));
  const denseByName = new Map(dense.map((match) => [match.name, match]));
  const normalizedTask = task.normalize('NFKC').toLocaleLowerCase();
  const fused = index.entries.map((entry): SkillsIndexMatch => {
    const denseMatch = denseByName.get(entry.name);
    const denseRank = denseRanks.get(entry.name);
    const lexicalRank = lexicalRanks.get(entry.name);
    const exactName = containsExplicitName(normalizedTask, entry.name);
    const score =
      (denseRank ? parameters.denseWeight / (parameters.rrfK + denseRank) : 0) +
      (lexicalRank ? parameters.lexicalWeight / (parameters.rrfK + lexicalRank) : 0);
    return {
      name: entry.name,
      description: entry.description,
      score: exactName ? Number.POSITIVE_INFINITY : score,
      denseScore: denseMatch?.score,
      lexicalScore: lexicalScores.get(entry.name) ?? 0,
      exactName,
      matchedChunk: denseMatch?.matchedChunk ?? 0,
      matchedSource: denseMatch?.matchedSource ?? 'document',
    };
  });
  fused.sort(
    (left, right) =>
      Number(right.exactName) - Number(left.exactName) ||
      right.score - left.score ||
      left.name.localeCompare(right.name),
  );
  const threshold = options.abstentionThreshold ?? parameters.abstentionThreshold;
  if (!fused[0]?.exactName && (fused[0]?.score ?? 0) < threshold) return [];
  return fused.slice(0, limit);
}

/** Rank with an already-computed query vector using frontmatter-guided MaxSim. */
export function rankSkillsIndex(
  index: SkillsIndex,
  query: ArrayLike<number>,
  options: Pick<SearchSkillsIndexOptions, 'limit' | 'minScore'> = {},
): readonly SkillsIndexMatch[] {
  if (!index.metadata.indexed) return [];
  const dimensions = index.metadata.dimensions;
  if (dimensions == null) throw new Error('Indexed skills metadata is missing dimensions');
  if (query.length !== dimensions) {
    throw new Error(`Query embedding has ${query.length} dimensions; expected ${dimensions}`);
  }
  const limit = positiveInteger(options.limit ?? index.metadata.retrievalLimit, 'limit');
  const minScore = options.minScore ?? Number.NEGATIVE_INFINITY;
  if (Number.isNaN(minScore)) throw new Error('minScore must be a number');

  return index.entries
    .map((entry) => ({
      name: entry.name,
      description: entry.description,
      ...scoreSkillsIndexEntry(entry, query),
    }))
    .filter((entry) => entry.score >= minScore)
    .sort((left, right) => right.score - left.score || left.name.localeCompare(right.name))
    .slice(0, limit);
}

/**
 * The first raw document chunk contains routing frontmatter. Later chunks
 * contribute supporting evidence without allowing long generic bodies to
 * dominate candidate generation.
 */
export function scoreSkillsIndexEntry(
  entry: SkillsIndexEntry,
  query: ArrayLike<number>,
): SkillsIndexEntryScore {
  if (entry.chunks.length === 0) throw new Error(`Skill '${entry.name}' has no chunks`);
  let best: SkillsIndexEntryScore | undefined;
  let firstChunkScore: number | undefined;
  for (let chunkIndex = 0; chunkIndex < entry.chunks.length; chunkIndex++) {
    const chunk = entry.chunks[chunkIndex];
    if (!chunk) continue;
    const score = skillVectorSimilarity(query, chunk.embedding);
    firstChunkScore ??= score;
    if (!best || score > best.score) {
      best = { score, matchedChunk: chunkIndex, matchedSource: chunk.source };
    }
  }
  if (!best) throw new Error(`Skill '${entry.name}' has no scoreable chunks`);
  if (firstChunkScore == null) throw new Error(`Skill '${entry.name}' has no first chunk`);
  return {
    ...best,
    score:
      SKILLS_INDEX_FIRST_CHUNK_WEIGHT * firstChunkScore +
      (1 - SKILLS_INDEX_FIRST_CHUNK_WEIGHT) * best.score,
  };
}

export function cosineSimilarity(left: ArrayLike<number>, right: ArrayLike<number>): number {
  if (left.length !== right.length) {
    throw new Error(`Cannot compare vectors with dimensions ${left.length} and ${right.length}`);
  }
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index++) {
    const a = Number(left[index]);
    const b = Number(right[index]);
    dot += a * b;
    leftNorm += a * a;
    rightNorm += b * b;
  }
  if (leftNorm === 0 || rightNorm === 0) return 0;
  return dot / Math.sqrt(leftNorm * rightNorm);
}

function skillVectorSimilarity(
  query: ArrayLike<number>,
  vector: Float32Array | QuantizedSkillVector,
): number {
  if (vector instanceof Float32Array) return cosineSimilarity(query, vector);
  if (query.length !== vector.values.length) {
    throw new Error(
      `Cannot compare vectors with dimensions ${query.length} and ${vector.values.length}`,
    );
  }
  let dot = 0;
  let queryNorm = 0;
  for (let index = 0; index < query.length; index++) {
    const value = Number(query[index]);
    dot += value * (vector.values[index] ?? 0);
    queryNorm += value * value;
  }
  if (queryNorm === 0 || vector.norm === 0) return 0;
  return (dot * vector.scale) / (Math.sqrt(queryNorm) * vector.norm);
}

function adapterMetadata(index: SkillsIndex): SkillVectorIndexMetadata {
  return {
    indexVersion: `${index.metadata.kind}@${index.metadata.version}:${index.metadata.catalogHash}`,
    catalogVersion: index.metadata.catalogHash,
    ready: index.metadata.indexed,
    dimensions: index.metadata.dimensions ?? 0,
    model: index.metadata.model,
    revision: index.metadata.revision,
    embedderId: index.metadata.embedderId,
    scoring: index.metadata.scoring,
  };
}

export function toSkillIndexWriteRequest(index: SkillsIndex): SkillIndexWriteRequest {
  const metadata = adapterMetadata(index);
  return {
    metadata: {
      indexVersion: metadata.indexVersion,
      catalogVersion: metadata.catalogVersion,
      dimensions: metadata.dimensions,
      model: metadata.model,
      revision: metadata.revision,
      embedderId: metadata.embedderId,
      scoring: metadata.scoring,
    },
    vectors: index.entries.flatMap((entry) =>
      entry.chunks.map((chunk, chunkIndex) => ({
        name: entry.name,
        description: entry.description,
        text: chunk.text,
        documentHash: entry.documentHash,
        chunk: chunkIndex,
        source: chunk.source,
        embedding: materializeVector(chunk.embedding),
      })),
    ),
  };
}

function groupWriteVectors(request: SkillIndexWriteRequest): SkillsIndexEntry[] {
  const grouped = new Map<string, SkillsIndexEntry>();
  for (const vector of request.vectors) {
    const existing = grouped.get(vector.name);
    const chunk = { source: vector.source, embedding: Float32Array.from(vector.embedding) };
    if (existing) {
      (existing.chunks as SkillsIndexChunk[]).push(chunk);
    } else {
      grouped.set(vector.name, {
        kind: 'skill',
        name: vector.name,
        description: vector.description,
        documentHash: vector.documentHash ?? '',
        chunks: [chunk],
      });
    }
  }
  return [...grouped.values()].sort((left, right) => left.name.localeCompare(right.name));
}

function buildLexicalIndex(skills: readonly AgentSkill[]): SkillsLexicalIndex {
  const postingMaps = new Map<string, Map<number, number>>();
  const documentLengths: number[] = [];
  for (let document = 0; document < skills.length; document++) {
    const skill = skills[document];
    if (!skill) continue;
    const weighted = [
      ...tokenizeLexical(skill.name).flatMap((token) => [token, token, token, token]),
      ...tokenizeLexical(skill.description ?? '').flatMap((token) => [token, token]),
      ...tokenizeLexical(skill.source),
    ];
    documentLengths.push(weighted.length);
    for (const token of weighted) {
      let posting = postingMaps.get(token);
      if (!posting) {
        posting = new Map();
        postingMaps.set(token, posting);
      }
      posting.set(document, (posting.get(document) ?? 0) + 1);
    }
  }
  const postings: Record<string, readonly number[]> = {};
  for (const [token, posting] of [...postingMaps].sort(([left], [right]) =>
    left.localeCompare(right),
  )) {
    postings[token] = [...posting].flatMap(([document, frequency]) => [document, frequency]);
  }
  return {
    documentLengths,
    averageDocumentLength:
      documentLengths.reduce((total, length) => total + length, 0) /
      Math.max(1, documentLengths.length),
    postings,
  };
}

function scoreLexicalIndex(
  index: SkillsLexicalIndex,
  task: string,
  parameters: SkillsIndexScoringParameters,
): ReadonlyMap<number, number> {
  const scores = new Map<number, number>();
  const queryTokens = new Set(tokenizeLexical(task));
  const documentCount = index.documentLengths.length;
  for (const token of queryTokens) {
    const posting = index.postings[token];
    if (!posting) continue;
    const documentFrequency = posting.length / 2;
    const inverseDocumentFrequency = Math.log(
      1 + (documentCount - documentFrequency + 0.5) / (documentFrequency + 0.5),
    );
    for (let offset = 0; offset < posting.length; offset += 2) {
      const document = posting[offset];
      const frequency = posting[offset + 1];
      if (document == null || frequency == null) continue;
      const length = index.documentLengths[document] ?? 0;
      const denominator =
        frequency +
        parameters.bm25K1 *
          (1 - parameters.bm25B + parameters.bm25B * (length / index.averageDocumentLength));
      const contribution =
        inverseDocumentFrequency * ((frequency * (parameters.bm25K1 + 1)) / denominator);
      scores.set(document, (scores.get(document) ?? 0) + contribution);
    }
  }
  return scores;
}

function tokenizeLexical(text: string): string[] {
  return (
    text
      .normalize('NFKC')
      .toLocaleLowerCase()
      .match(/[\p{L}\p{N}_-]+/gu)
      ?.filter((token) => token.length > 2) ?? []
  );
}

function containsExplicitName(normalizedTask: string, name: string): boolean {
  const normalizedName = name.normalize('NFKC').toLocaleLowerCase();
  let start = normalizedTask.indexOf(normalizedName);
  while (start >= 0) {
    const before = start === 0 ? '' : (normalizedTask[start - 1] ?? '');
    const after = normalizedTask[start + normalizedName.length] ?? '';
    if (!/[\p{L}\p{N}_-]/u.test(before) && !/[\p{L}\p{N}_-]/u.test(after)) return true;
    start = normalizedTask.indexOf(normalizedName, start + 1);
  }
  return false;
}

function parseLexicalIndex(value: unknown, skillCount: number, file: string): SkillsLexicalIndex {
  const lexical = value as Partial<SkillsLexicalIndex> | undefined;
  if (
    !lexical ||
    !Array.isArray(lexical.documentLengths) ||
    lexical.documentLengths.length !== skillCount ||
    lexical.documentLengths.some((length) => !isNonNegativeInteger(length)) ||
    typeof lexical.averageDocumentLength !== 'number' ||
    !Number.isFinite(lexical.averageDocumentLength) ||
    lexical.averageDocumentLength <= 0 ||
    !lexical.postings ||
    typeof lexical.postings !== 'object'
  ) {
    throw new Error(`Invalid skills lexical index: ${file}`);
  }
  for (const [token, posting] of Object.entries(lexical.postings)) {
    if (
      !token ||
      !Array.isArray(posting) ||
      posting.length % 2 !== 0 ||
      posting.some((number) => !Number.isFinite(number))
    ) {
      throw new Error(`Invalid lexical posting '${token}' in ${file}`);
    }
    for (let offset = 0; offset < posting.length; offset += 2) {
      if (
        !isNonNegativeInteger(posting[offset]) ||
        (posting[offset] ?? skillCount) >= skillCount ||
        !isPositiveInteger(posting[offset + 1])
      ) {
        throw new Error(`Invalid lexical posting '${token}' in ${file}`);
      }
    }
  }
  return lexical as SkillsLexicalIndex;
}

export function hashSkillCatalog(skills: readonly Pick<AgentSkill, 'name' | 'source'>[]): string {
  const sorted = skills
    .map((skill) => ({
      name: skill.name,
      documentHash: hashText(skillIndexText(skill)),
    }))
    .sort((left, right) => left.name.localeCompare(right.name));
  const hash = createHash('sha256');
  for (const skill of sorted) hash.update(`${JSON.stringify(skill)}\n`);
  return hash.digest('hex');
}

function assertCompatibleEmbedder(index: SkillsIndex, embedder: SkillEmbedder): void {
  if (
    embedder.id !== index.metadata.embedderId ||
    embedder.model !== index.metadata.model ||
    embedder.revision !== index.metadata.revision
  ) {
    throw new Error(
      `Skills index uses ${index.metadata.embedderId}, but the query embedder uses ${embedder.id}`,
    );
  }
}

function normalizeVector(input: ArrayLike<number>): Float32Array {
  let norm = 0;
  for (let index = 0; index < input.length; index++) {
    const value = Number(input[index]);
    if (!Number.isFinite(value)) throw new Error('Skill embedder returned a non-finite value');
    norm += value * value;
  }
  if (norm === 0) throw new Error('Skill embedder returned a zero vector');
  const scale = 1 / Math.sqrt(norm);
  return Float32Array.from(input, (value) => Number(value) * scale);
}

interface V3ChunkRecord {
  readonly source: SkillsIndexChunkSource;
  readonly offset: number;
  readonly scale: number;
  readonly norm: number;
}

interface V3EntryRecord {
  readonly kind: 'skill';
  readonly name: string;
  readonly description: string;
  readonly documentHash: string;
  readonly chunks: readonly V3ChunkRecord[];
}

function writeV3IndexAtomically(file: string, index: SkillsIndex): void {
  mkdirSync(dirname(file), { recursive: true });
  if (!index.metadata.indexed) {
    writeJsonAtomically(file, {
      metadata: { ...index.metadata, version: SKILLS_INDEX_VERSION },
      entries: [],
    });
    return;
  }
  const dimensions = index.metadata.dimensions;
  if (!dimensions) throw new Error('Indexed skills metadata is missing dimensions');
  const vectorFile = `${file}.vectors.bin`;
  const vectorTemporary = uniqueTemporaryPath(vectorFile);
  const chunks: Buffer[] = [];
  let offset = 0;
  const entries: V3EntryRecord[] = index.entries.map((entry) => ({
    kind: 'skill',
    name: entry.name,
    description: entry.description,
    documentHash: entry.documentHash,
    chunks: entry.chunks.map((chunk) => {
      const quantized = quantizeVector(chunk.embedding);
      if (quantized.values.length !== dimensions) {
        throw new Error(
          `Skill '${entry.name}' vector has ${quantized.values.length} dimensions; expected ${dimensions}`,
        );
      }
      const bytes = Buffer.from(
        quantized.values.buffer,
        quantized.values.byteOffset,
        quantized.values.byteLength,
      );
      chunks.push(bytes);
      const record = {
        source: chunk.source,
        offset,
        scale: quantized.scale,
        norm: quantized.norm,
      };
      offset += bytes.length;
      return record;
    }),
  }));
  const vectors = Buffer.concat(chunks);
  const metadata: SkillsIndexMetadata = {
    ...index.metadata,
    version: SKILLS_INDEX_VERSION,
    scoring: SKILLS_INDEX_SCORING,
    scoringParameters: DEFAULT_SKILLS_INDEX_SCORING_PARAMETERS,
    vectorEncoding: SKILLS_INDEX_VECTOR_ENCODING,
    vectorFile: basename(vectorFile),
    vectorHash: hashBytes(vectors),
    vectorBytes: vectors.length,
  };
  try {
    writePrivateFile(vectorTemporary, vectors);
    renameSync(vectorTemporary, vectorFile);
    writeJsonAtomically(file, { metadata, entries, lexical: index.lexical });
  } catch (error) {
    try {
      unlinkSync(vectorTemporary);
    } catch {
      // The temporary sidecar may already have been published.
    }
    throw error;
  }
}

function quantizeVector(vector: Float32Array | QuantizedSkillVector): QuantizedSkillVector {
  if (!(vector instanceof Float32Array)) return vector;
  let maximum = 0;
  for (const value of vector) {
    maximum = Math.max(maximum, Math.abs(value));
  }
  if (maximum === 0) throw new Error('Cannot quantize a zero vector');
  const scale = maximum / 127;
  const values = Int8Array.from(vector, (value) =>
    Math.max(-127, Math.min(127, Math.round(value / scale))),
  );
  let quantizedNormSquared = 0;
  for (const value of values) quantizedNormSquared += (value * scale) ** 2;
  return { values, scale, norm: Math.sqrt(quantizedNormSquared) };
}

function parseV3Index(
  value: { metadata?: Partial<SkillsIndexMetadata>; entries?: unknown; lexical?: unknown },
  file: string,
): SkillsIndex {
  const metadata = parseMetadata(JSON.stringify(value.metadata), file, SKILLS_INDEX_VERSION);
  if (!Array.isArray(value.entries)) throw new Error(`Invalid skills index entries: ${file}`);
  if (!metadata.indexed) {
    if (value.entries.length !== 0)
      throw new Error(`Non-indexed skills artifact contains entries: ${file}`);
    return { file, metadata, entries: [] };
  }
  if (!metadata.vectorFile || !metadata.vectorHash || metadata.vectorBytes == null) {
    throw new Error(`Indexed skills metadata is missing vector sidecar information: ${file}`);
  }
  const sidecar = resolve(dirname(file), metadata.vectorFile);
  let bytes: Buffer;
  try {
    bytes = readFileSync(sidecar);
  } catch (error) {
    throw new Error(`Skills index vector sidecar is missing: ${sidecar}`, { cause: error });
  }
  if (statSync(sidecar).size !== metadata.vectorBytes || bytes.length !== metadata.vectorBytes) {
    throw new Error(`Skills index vector sidecar is truncated: ${sidecar}`);
  }
  if (hashBytes(bytes) !== metadata.vectorHash) {
    throw new Error(`Skills index vector sidecar hash mismatch: ${sidecar}`);
  }
  const dimensions = metadata.dimensions ?? 0;
  if (bytes.length !== metadata.chunkCount * dimensions) {
    throw new Error(`Skills index vector sidecar dimension mismatch: ${sidecar}`);
  }
  const names = new Set<string>();
  let chunkCount = 0;
  const entries = value.entries.map((raw, entryIndex): SkillsIndexEntry => {
    const entry = raw as Partial<V3EntryRecord>;
    if (
      entry.kind !== 'skill' ||
      typeof entry.name !== 'string' ||
      typeof entry.description !== 'string' ||
      typeof entry.documentHash !== 'string' ||
      !Array.isArray(entry.chunks)
    ) {
      throw new Error(`Invalid skill index entry ${entryIndex} in ${file}`);
    }
    if (names.has(entry.name)) throw new Error(`Duplicate skill '${entry.name}' in ${file}`);
    names.add(entry.name);
    const chunks = entry.chunks.map((rawChunk): SkillsIndexChunk => {
      const chunk = rawChunk as Partial<V3ChunkRecord>;
      if (
        chunk.source !== 'document' ||
        !isNonNegativeInteger(chunk.offset) ||
        typeof chunk.scale !== 'number' ||
        !Number.isFinite(chunk.scale) ||
        chunk.scale <= 0 ||
        typeof chunk.norm !== 'number' ||
        !Number.isFinite(chunk.norm) ||
        chunk.norm <= 0 ||
        chunk.offset + dimensions > bytes.length
      ) {
        throw new Error(`Invalid skill vector reference for '${entry.name}' in ${file}`);
      }
      chunkCount++;
      return {
        source: chunk.source,
        embedding: {
          values: new Int8Array(bytes.buffer, bytes.byteOffset + chunk.offset, dimensions),
          scale: chunk.scale,
          norm: chunk.norm,
        },
      };
    });
    if (chunks.length === 0) throw new Error(`Skill '${entry.name}' has no chunks`);
    return {
      kind: 'skill',
      name: entry.name,
      description: entry.description,
      documentHash: entry.documentHash,
      chunks,
    };
  });
  if (entries.length !== metadata.skillCount || chunkCount !== metadata.chunkCount) {
    throw new Error(`Skills index manifest counts do not match metadata: ${file}`);
  }
  const lexical =
    value.lexical == null ? undefined : parseLexicalIndex(value.lexical, entries.length, file);
  return { file, metadata, entries, lexical };
}

function hashBytes(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function uniqueTemporaryPath(file: string): string {
  return `${file}.${process.pid}.${randomUUID()}.tmp`;
}

function writePrivateFile(file: string, data: string | NodeJS.ArrayBufferView): void {
  writeFileSync(file, data, { mode: 0o600, flag: 'wx' });
}

function writeJsonAtomically(file: string, value: unknown): void {
  mkdirSync(dirname(file), { recursive: true });
  const temporary = uniqueTemporaryPath(file);
  try {
    writePrivateFile(temporary, `${JSON.stringify(value)}\n`);
    renameSync(temporary, file);
  } catch (error) {
    try {
      unlinkSync(temporary);
    } catch {
      // The temporary file may not have been created.
    }
    throw error;
  }
}

function materializeVector(vector: Float32Array | QuantizedSkillVector): Float32Array {
  if (vector instanceof Float32Array) return vector;
  return Float32Array.from(vector.values, (value) => value * vector.scale);
}

function decodeVector(
  value: string,
  dimensions: number,
  file: string,
  lineNumber: number,
): Float32Array {
  if (value.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(value)) {
    throw new Error(`Invalid base64 vector at ${file}:${lineNumber}`);
  }
  const bytes = Buffer.from(value, 'base64');
  const expectedBytes = dimensions * Float32Array.BYTES_PER_ELEMENT;
  if (bytes.length !== expectedBytes) {
    throw new Error(
      `Invalid vector length at ${file}:${lineNumber}; got ${bytes.length} bytes, expected ${expectedBytes}`,
    );
  }
  const vector = new Float32Array(dimensions);
  for (let index = 0; index < dimensions; index++) {
    const number = bytes.readFloatLE(index * Float32Array.BYTES_PER_ELEMENT);
    if (!Number.isFinite(number)) {
      throw new Error(`Non-finite vector value at ${file}:${lineNumber}`);
    }
    vector[index] = number;
  }
  return vector;
}

function parseMetadata(
  line: string,
  file: string,
  expectedVersion: 2 | typeof SKILLS_INDEX_VERSION,
): SkillsIndexMetadata {
  const value = parseLine(line, 1, file) as Partial<SkillsIndexMetadata>;
  if (value.kind !== SKILLS_INDEX_FORMAT || value.version !== expectedVersion) {
    throw unsupportedIndexError(file);
  }
  if (
    typeof value.indexed !== 'boolean' ||
    !isNonNegativeInteger(value.skillCount) ||
    !isNonNegativeInteger(value.chunkCount) ||
    !isNonNegativeInteger(value.threshold) ||
    !isPositiveInteger(value.retrievalLimit) ||
    !isPositiveInteger(value.chunkTokens) ||
    !isNonNegativeInteger(value.chunkOverlapTokens) ||
    value.chunkOverlapTokens >= value.chunkTokens ||
    value.scoring !== (expectedVersion === 2 ? SKILLS_INDEX_V2_SCORING : SKILLS_INDEX_SCORING) ||
    value.vectorEncoding !==
      (expectedVersion === 2 ? SKILLS_INDEX_V2_VECTOR_ENCODING : SKILLS_INDEX_VECTOR_ENCODING) ||
    typeof value.catalogHash !== 'string'
  ) {
    throw new Error(`Invalid skills index metadata: ${file}`);
  }
  if (
    expectedVersion === SKILLS_INDEX_VERSION &&
    (!value.scoringParameters ||
      !Number.isFinite(value.scoringParameters.denseWeight) ||
      !Number.isFinite(value.scoringParameters.lexicalWeight) ||
      !Number.isFinite(value.scoringParameters.rrfK) ||
      value.scoringParameters.rrfK <= 0 ||
      !Number.isFinite(value.scoringParameters.bm25K1) ||
      value.scoringParameters.bm25K1 <= 0 ||
      !Number.isFinite(value.scoringParameters.bm25B) ||
      value.scoringParameters.bm25B < 0 ||
      value.scoringParameters.bm25B > 1 ||
      !Number.isFinite(value.scoringParameters.abstentionThreshold) ||
      value.scoringParameters.abstentionThreshold < 0)
  ) {
    throw new Error(`Invalid skills index scoring parameters: ${file}`);
  }
  if (
    value.indexed &&
    (typeof value.model !== 'string' ||
      typeof value.revision !== 'string' ||
      typeof value.embedderId !== 'string' ||
      !isPositiveInteger(value.dimensions) ||
      value.chunkCount < value.skillCount)
  ) {
    throw new Error(`Indexed skills metadata is missing model or chunk information: ${file}`);
  }
  if (!value.indexed && value.chunkCount !== 0) {
    throw new Error(`Non-indexed skills metadata declares chunks: ${file}`);
  }
  return value as SkillsIndexMetadata;
}

function unsupportedIndexError(file: string): Error {
  return new Error(
    `Unsupported skills index format or version: ${file}. Run: di-framework skills index migrate --input "${file}" --output "${file}"`,
  );
}

function parseEntry(
  line: string,
  lineNumber: number,
  file: string,
  dimensions?: number,
): SkillsIndexEntry {
  const value = parseLine(line, lineNumber, file) as {
    kind?: unknown;
    name?: unknown;
    description?: unknown;
    documentHash?: unknown;
    chunks?: unknown;
  };
  if (
    value.kind !== 'skill' ||
    typeof value.name !== 'string' ||
    typeof value.description !== 'string' ||
    typeof value.documentHash !== 'string' ||
    !Array.isArray(value.chunks) ||
    value.chunks.length === 0 ||
    dimensions == null
  ) {
    throw new Error(`Invalid skill index entry at ${file}:${lineNumber}`);
  }
  const chunks = value.chunks.map((raw) => {
    const chunk = raw as { source?: unknown; vector?: unknown };
    if (chunk.source !== 'document' || typeof chunk.vector !== 'string') {
      throw new Error(`Invalid skill chunk at ${file}:${lineNumber}`);
    }
    const source: SkillsIndexChunkSource = chunk.source;
    return {
      source,
      embedding: decodeVector(chunk.vector, dimensions, file, lineNumber),
    };
  });
  return {
    kind: 'skill',
    name: value.name,
    description: value.description,
    documentHash: value.documentHash,
    chunks,
  };
}

function parseLine(line: string, lineNumber: number, file: string): unknown {
  try {
    return JSON.parse(line);
  } catch (error) {
    throw new Error(`Invalid JSON at ${file}:${lineNumber}`, { cause: error });
  }
}

function hashText(text: string): string {
  return createHash('sha256').update(text).digest('hex');
}

function staleIndexError(index: SkillsIndex, detail: string): Error {
  return new Error(
    `Skills index is stale (${detail}). Rebuild ${index.file ?? DEFAULT_SKILLS_INDEX_FILE}`,
  );
}

function positiveInteger(value: number, name: string): number {
  if (!isPositiveInteger(value)) throw new Error(`${name} must be a positive integer`);
  return value;
}

function nonNegativeInteger(value: number, name: string): number {
  if (!isNonNegativeInteger(value)) throw new Error(`${name} must be a non-negative integer`);
  return value;
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0;
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}
