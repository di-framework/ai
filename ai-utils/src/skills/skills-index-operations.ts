import { statSync } from 'node:fs';
import { resolve } from 'node:path';
import { expandUserPath } from '../sandbox/paths.ts';
import type { AgentSkill } from './parse-skill-markdown.ts';
import type { SkillEmbedder } from './skill-embedder.ts';
import { TransformersJsSkillEmbedder } from './skill-embedder.ts';
import {
  assertSkillsIndexCurrent,
  DEFAULT_SKILLS_INDEX_FILE,
  LocalSkillIndexWriter,
  loadSkillsIndex,
  rankHybridSkillsIndex,
  SKILLS_INDEX_VERSION,
  type SkillsIndexMatch,
  type SkillsIndexMetadata,
} from './skills-index.ts';
import {
  asSkillsIndexOperationError,
  type SkillsIndexOperation,
  SkillsIndexOperationError,
} from './skills-index-errors.ts';
import { collectSkills } from './skills-tool.ts';

export type SkillsIndexOperationPhase =
  | 'load'
  | 'collect-sources'
  | 'embed-query'
  | 'search'
  | 'write';

export interface SkillsIndexOperationProgress {
  readonly operation: Exclude<SkillsIndexOperation, 'build'>;
  readonly phase: SkillsIndexOperationPhase;
  readonly completed: number;
  readonly total: number;
}

export type SkillsIndexOperationProgressCallback = (progress: SkillsIndexOperationProgress) => void;

interface SkillsIndexFileOperationOptions {
  readonly inputFile?: string;
  readonly cwd?: string;
  readonly onProgress?: SkillsIndexOperationProgressCallback;
}

export type InspectSkillsIndexOptions = SkillsIndexFileOperationOptions;

export interface InspectSkillsIndexResult {
  readonly file: string;
  readonly metadata: SkillsIndexMetadata;
  readonly manifestBytes: number;
  readonly vectorBytes: number;
  readonly lexicalTerms: number;
  readonly loadMs: number;
  readonly rssBytes: number;
}

export interface ValidateSkillsIndexOptions extends SkillsIndexFileOperationOptions {
  readonly skills?: readonly AgentSkill[];
  readonly directories?: readonly string[];
  readonly files?: readonly string[];
  readonly allowExtraSkills?: boolean;
}

export interface ValidateSkillsIndexResult {
  readonly file: string;
  readonly integrity: 'valid';
  readonly sourceDrift: string | null;
  readonly valid: boolean;
  readonly ready: boolean;
  readonly loadMs: number;
}

export interface QuerySkillsIndexOptions extends SkillsIndexFileOperationOptions {
  readonly query: string;
  readonly embedder?: SkillEmbedder;
  readonly limit?: number;
  readonly minScore?: number;
  readonly abstentionThreshold?: number;
}

export interface QuerySkillsIndexTimings {
  readonly loadMs: number;
  readonly embedMs: number;
  readonly searchMs: number;
  readonly totalMs: number;
}

export interface QuerySkillsIndexResult {
  readonly file: string;
  readonly decision: 'selected' | 'abstained';
  readonly matches: readonly SkillsIndexMatch[];
  readonly timings: QuerySkillsIndexTimings;
  readonly rssBytes: number;
}

export interface MigrateSkillsIndexOptions extends SkillsIndexFileOperationOptions {
  readonly outputFile?: string;
}

export interface MigrateSkillsIndexResult {
  readonly inputFile: string;
  readonly outputFile: string;
  readonly fromVersion: SkillsIndexMetadata['version'];
  readonly toVersion: typeof SKILLS_INDEX_VERSION;
  readonly loadMs: number;
}

/** Load an index and return safe metadata without skill bodies or vector contents. */
export function inspectSkillsIndex(
  options: InspectSkillsIndexOptions = {},
): InspectSkillsIndexResult {
  const started = performance.now();
  const rssBefore = process.memoryUsage.rss();
  const file = operationPath(options.inputFile, options.cwd);
  const index = loadForOperation('inspect', file);
  progress(options, 'inspect', 'load');
  try {
    return {
      file,
      metadata: index.metadata,
      manifestBytes: statSync(file).size,
      vectorBytes: index.metadata.vectorBytes ?? 0,
      lexicalTerms: Object.keys(index.lexical?.postings ?? {}).length,
      loadMs: performance.now() - started,
      rssBytes: process.memoryUsage.rss() - rssBefore,
    };
  } catch (error) {
    throw asSkillsIndexOperationError('inspect', 'OPERATION_FAILED', error);
  }
}

/** Verify index integrity and optionally compare it with explicit skill sources. */
export function validateSkillsIndex(
  options: ValidateSkillsIndexOptions = {},
): ValidateSkillsIndexResult {
  const started = performance.now();
  const file = operationPath(options.inputFile, options.cwd);
  const index = loadForOperation('validate', file);
  progress(options, 'validate', 'load');
  const hasSources =
    (options.skills?.length ?? 0) > 0 ||
    (options.directories?.length ?? 0) > 0 ||
    (options.files?.length ?? 0) > 0;
  let sourceDrift: string | null = null;
  if (hasSources) {
    let skills: readonly AgentSkill[];
    try {
      skills = collectSkills({
        skills: options.skills,
        directories: resolveSources(options.directories, options.cwd),
        files: resolveSources(options.files, options.cwd),
      });
      progress(options, 'validate', 'collect-sources');
    } catch (error) {
      throw asSkillsIndexOperationError('validate', 'SOURCE_NOT_FOUND', error);
    }
    try {
      assertSkillsIndexCurrent(index, skills, { allowExtraSkills: options.allowExtraSkills });
    } catch (error) {
      sourceDrift = error instanceof Error ? error.message : String(error);
    }
  }
  return {
    file,
    integrity: 'valid',
    sourceDrift,
    valid: sourceDrift == null,
    ready: index.metadata.indexed,
    loadMs: performance.now() - started,
  };
}

/** Embed a query and return safe hybrid matches from an existing index. */
export async function querySkillsIndex(
  options: QuerySkillsIndexOptions,
): Promise<QuerySkillsIndexResult> {
  if (!options.query?.trim()) {
    throw new SkillsIndexOperationError('query', 'INVALID_OPTIONS', 'query must not be empty');
  }
  const started = performance.now();
  const rssBefore = process.memoryUsage.rss();
  const file = operationPath(options.inputFile, options.cwd);
  const loadStarted = performance.now();
  const index = loadForOperation('query', file);
  const loadMs = performance.now() - loadStarted;
  progress(options, 'query', 'load');
  const embedder =
    options.embedder ??
    new TransformersJsSkillEmbedder({
      model: index.metadata.model,
      revision: index.metadata.revision,
    });
  const embedStarted = performance.now();
  let vector: Float32Array | undefined;
  try {
    [vector] = await embedder.embed([options.query], { purpose: 'query' });
    if (!vector) throw new Error('Skill embedder did not return a query vector');
  } catch (error) {
    throw asSkillsIndexOperationError('query', 'EMBEDDING_FAILED', error);
  }
  const embedMs = performance.now() - embedStarted;
  progress(options, 'query', 'embed-query');
  const searchStarted = performance.now();
  let matches: readonly SkillsIndexMatch[];
  try {
    matches = rankHybridSkillsIndex(index, vector, options.query, {
      limit: options.limit,
      minScore: options.minScore,
      abstentionThreshold: options.abstentionThreshold,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw asSkillsIndexOperationError(
      'query',
      /dimension|vector/i.test(message) ? 'EMBEDDING_FAILED' : 'INVALID_OPTIONS',
      error,
    );
  }
  const searchMs = performance.now() - searchStarted;
  progress(options, 'query', 'search');
  return {
    file,
    decision: matches.length > 0 ? 'selected' : 'abstained',
    matches,
    timings: { loadMs, embedMs, searchMs, totalMs: performance.now() - started },
    rssBytes: process.memoryUsage.rss() - rssBefore,
  };
}

/** Rewrite a readable legacy or current index with the current local format. */
export function migrateSkillsIndex(
  options: MigrateSkillsIndexOptions = {},
): MigrateSkillsIndexResult {
  const inputFile = operationPath(options.inputFile, options.cwd);
  const outputFile = operationPath(options.outputFile, options.cwd);
  const started = performance.now();
  const index = loadForOperation('migrate', inputFile);
  const loadMs = performance.now() - started;
  progress(options, 'migrate', 'load');
  try {
    new LocalSkillIndexWriter(outputFile).writeIndex(index);
  } catch (error) {
    throw asSkillsIndexOperationError('migrate', 'WRITE_FAILED', error);
  }
  progress(options, 'migrate', 'write');
  return {
    inputFile,
    outputFile,
    fromVersion: index.metadata.version,
    toVersion: SKILLS_INDEX_VERSION,
    loadMs,
  };
}

function operationPath(path: string | undefined, cwd = process.cwd()): string {
  return resolve(cwd, expandUserPath(path ?? DEFAULT_SKILLS_INDEX_FILE));
}

function resolveSources(sources: readonly string[] | undefined, cwd = process.cwd()): string[] {
  return (sources ?? []).map((source) => resolve(cwd, expandUserPath(source)));
}

function loadForOperation(operation: Exclude<SkillsIndexOperation, 'build'>, file: string) {
  try {
    return loadSkillsIndex(file);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const code = message.startsWith('Skills index does not exist or cannot be read')
      ? 'INDEX_NOT_FOUND'
      : 'INVALID_INDEX';
    throw asSkillsIndexOperationError(operation, code, error);
  }
}

function progress(
  options: { readonly onProgress?: SkillsIndexOperationProgressCallback },
  operation: Exclude<SkillsIndexOperation, 'build'>,
  phase: SkillsIndexOperationPhase,
): void {
  options.onProgress?.({ operation, phase, completed: 1, total: 1 });
}
