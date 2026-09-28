import type { SkillVectorQueryOptions, SkillVectorSearch } from './skill-adapters.ts';

export interface SkillAdapterQualityMetrics {
  readonly positiveTrials: number;
  readonly noSkillTrials: number;
  readonly recallAt1: number;
  readonly recallAt10: number;
  readonly meanReciprocalRank: number;
  readonly abstentionRate: number;
  readonly noSkillFalsePositiveRate: number;
}

/** Quality identity compatible with #194's versioned RetrievalEvaluationResult. */
export interface SkillAdapterEvaluationQuality {
  readonly schemaVersion: 1;
  readonly suite: string;
  readonly corpus: {
    readonly id: string;
    readonly revision: string;
    readonly skillCount: number;
  };
  readonly trialsPerCase: number;
  readonly caseCount: number;
  readonly metrics: SkillAdapterQualityMetrics;
}

export type SkillAdapterQualityInput = SkillAdapterQualityMetrics | SkillAdapterEvaluationQuality;

export interface SkillAdapterBenchmarkCase {
  readonly vector: ArrayLike<number>;
  readonly options?: SkillVectorQueryOptions;
}

export interface SkillAdapterBenchmarkOptions {
  readonly createSearch: () => SkillVectorSearch | Promise<SkillVectorSearch>;
  readonly cases: readonly SkillAdapterBenchmarkCase[];
  readonly warmupTrials?: number;
  readonly measuredTrials?: number;
  readonly now?: () => number;
  readonly memoryBytes?: () => number;
  readonly network?: () => { readonly requests: number; readonly bytes: number };
  /** Metrics or the complete versioned identity from #194's evaluation result. */
  readonly quality?: SkillAdapterQualityInput;
  readonly indexBuildMs?: number;
  readonly readinessMs?: number;
  readonly artifactBytes?: number;
}

export interface SkillAdapterPerformanceReport {
  readonly schemaVersion: 1;
  readonly cases: number;
  readonly measuredTrials: number;
  readonly coldInitializationMs: number;
  readonly vectorSearchMs: {
    readonly p50: number;
    readonly p95: number;
    readonly mean: number;
  };
  readonly endToEndSelectionMs: {
    readonly p50: number;
    readonly p95: number;
    readonly mean: number;
  };
  readonly peakMemoryBytes?: number;
  readonly networkRequests?: number;
  readonly networkBytes?: number;
  readonly indexBuildMs?: number;
  readonly readinessMs?: number;
  readonly artifactBytes?: number;
  readonly quality?: SkillAdapterQualityInput;
}

/** Deterministic measurement shell shared by local and platform adapters. */
export async function benchmarkSkillVectorSearch(
  options: SkillAdapterBenchmarkOptions,
): Promise<SkillAdapterPerformanceReport> {
  if (options.cases.length === 0) throw new Error('At least one benchmark case is required');
  const warmupTrials = nonNegativeInteger(options.warmupTrials ?? 1, 'warmupTrials');
  const measuredTrials = positiveInteger(options.measuredTrials ?? 10, 'measuredTrials');
  const now = options.now ?? (() => performance.now());
  const initializationStart = now();
  const search = await options.createSearch();
  const coldInitializationMs = now() - initializationStart;
  let peakMemoryBytes = options.memoryBytes?.();

  for (let trial = 0; trial < warmupTrials; trial++) {
    for (const entry of options.cases) await search.query(entry.vector, entry.options);
  }

  const timings: number[] = [];
  for (let trial = 0; trial < measuredTrials; trial++) {
    for (const entry of options.cases) {
      const start = now();
      await search.query(entry.vector, entry.options);
      timings.push(now() - start);
      const memory = options.memoryBytes?.();
      if (memory != null) peakMemoryBytes = Math.max(peakMemoryBytes ?? memory, memory);
    }
  }
  const summary = summarize(timings);
  const network = options.network?.();
  return {
    schemaVersion: 1,
    cases: options.cases.length,
    measuredTrials,
    coldInitializationMs,
    vectorSearchMs: summary,
    endToEndSelectionMs: summary,
    peakMemoryBytes,
    networkRequests: network?.requests,
    networkBytes: network?.bytes,
    indexBuildMs: options.indexBuildMs,
    readinessMs: options.readinessMs,
    artifactBytes: options.artifactBytes,
    quality: normalizeQuality(options.quality),
  };
}

function normalizeQuality(
  quality?: SkillAdapterQualityInput,
): SkillAdapterQualityInput | undefined {
  if (quality == null) return undefined;
  if ('metrics' in quality) {
    return {
      schemaVersion: quality.schemaVersion,
      suite: quality.suite,
      corpus: { ...quality.corpus },
      trialsPerCase: quality.trialsPerCase,
      caseCount: quality.caseCount,
      metrics: { ...quality.metrics },
    };
  }
  return { ...quality };
}

function summarize(values: readonly number[]): { p50: number; p95: number; mean: number } {
  const sorted = values.slice().sort((left, right) => left - right);
  return {
    p50: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
    mean: sorted.reduce((total, value) => total + value, 0) / sorted.length,
  };
}

function percentile(sorted: readonly number[], fraction: number): number {
  return sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)] ?? 0;
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) throw new Error(`${name} must be a positive integer`);
  return value;
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative integer`);
  }
  return value;
}
