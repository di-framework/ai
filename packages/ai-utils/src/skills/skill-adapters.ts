import type { AgentSkill } from './parse-skill-markdown.ts';

/** Provider-neutral information safe to include in the initial skill catalog. */
export interface SkillDescriptor {
  readonly name: string;
  readonly description?: string;
  readonly sourceHash: string;
  readonly version?: string;
  readonly namespace?: string;
  readonly references?: readonly string[];
  readonly assets?: readonly string[];
}

export interface SkillAdapterCapabilities {
  readonly namespaces: boolean;
  readonly lazyBodies: boolean;
  readonly vectorSearch: boolean;
  readonly indexWriting: boolean;
  readonly eventuallyConsistent: boolean;
}

export type SkillAdapterHealthStatus = 'ready' | 'degraded' | 'not-ready';

export interface SkillAdapterHealth {
  readonly status: SkillAdapterHealthStatus;
  readonly message?: string;
  readonly checkedVersion?: string;
}

export type SkillAdapterErrorCode =
  | 'STALE_CATALOG'
  | 'MODEL_MISMATCH'
  | 'MISSING_BODY'
  | 'TIMEOUT'
  | 'PARTIAL_RESULT'
  | 'NOT_READY'
  | 'INVALID_RESPONSE';

export const DEFAULT_SKILL_ADAPTER_TIMEOUT_MS = 10_000;

/** A stable, typed error boundary for platform adapter failures. */
export class SkillAdapterError extends Error {
  override readonly name = 'SkillAdapterError';

  constructor(
    readonly code: SkillAdapterErrorCode,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
  }
}

/** Bound remote operations and normalize provider failures without fallback. */
export async function runSkillAdapterOperation<T>(
  operation: string,
  call: () => PromiseLike<T>,
  timeoutMs = DEFAULT_SKILL_ADAPTER_TIMEOUT_MS,
): Promise<T> {
  if (!Number.isInteger(timeoutMs) || timeoutMs < 1) {
    throw new SkillAdapterError('INVALID_RESPONSE', 'Adapter timeout must be a positive integer');
  }
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      Promise.resolve().then(call),
      new Promise<never>((_, reject) => {
        timer = setTimeout(
          () =>
            reject(
              new SkillAdapterError(
                'TIMEOUT',
                `${operation} timed out after ${timeoutMs} milliseconds`,
              ),
            ),
          timeoutMs,
        );
      }),
    ]);
  } catch (error) {
    if (error instanceof SkillAdapterError) throw error;
    throw new SkillAdapterError('INVALID_RESPONSE', `${operation} failed`, { cause: error });
  } finally {
    if (timer != null) clearTimeout(timer);
  }
}

export interface SkillCatalogListOptions {
  readonly namespace?: string;
}

export interface SkillCatalogLoadOptions extends SkillCatalogListOptions {
  /** Reject a body whose descriptor no longer has this version. */
  readonly expectedVersion?: string;
}

export interface SkillCatalogStore {
  readonly capabilities: SkillAdapterCapabilities;
  list(options?: SkillCatalogListOptions): Promise<readonly SkillDescriptor[]>;
  load(name: string, options?: SkillCatalogLoadOptions): Promise<AgentSkill | undefined>;
  version(options?: SkillCatalogListOptions): Promise<string>;
  health(options?: SkillCatalogListOptions): Promise<SkillAdapterHealth>;
}

export type SkillChunkSource = 'document';

/** A backend-neutral chunk result. Skill-level aggregation remains shared. */
export interface SkillChunkMatch {
  readonly name: string;
  readonly description: string;
  readonly score: number;
  readonly chunk: number;
  readonly source: SkillChunkSource;
}

export interface SkillVectorQueryOptions {
  readonly limit?: number;
  readonly minScore?: number;
  readonly namespace?: string;
  readonly catalogVersion?: string;
  readonly indexVersion?: string;
  readonly model?: string;
  readonly revision?: string;
  readonly embedderId?: string;
}

export interface SkillVectorIndexMetadata {
  readonly indexVersion: string;
  readonly catalogVersion: string;
  readonly ready: boolean;
  readonly dimensions: number;
  readonly model?: string;
  readonly revision?: string;
  readonly embedderId?: string;
  readonly namespace?: string;
  readonly scoring: string;
}

export interface SkillVectorSearch {
  readonly capabilities: SkillAdapterCapabilities;
  metadata(options?: SkillCatalogListOptions): Promise<SkillVectorIndexMetadata>;
  query(
    vector: ArrayLike<number>,
    options?: SkillVectorQueryOptions,
  ): Promise<readonly SkillChunkMatch[]>;
  health(options?: SkillCatalogListOptions): Promise<SkillAdapterHealth>;
}

export interface SkillIndexVector {
  readonly name: string;
  readonly description: string;
  /**
   * Exact source chunk used to produce the embedding. Writers may persist it as
   * the vector document text, but search results and discovery prompts must not
   * expose it.
   */
  readonly text?: string;
  readonly chunk: number;
  readonly source: SkillChunkSource;
  /** Optional source identity retained by deterministic artifact writers. */
  readonly documentHash?: string;
  readonly embedding: ArrayLike<number>;
}

export interface SkillIndexWriteRequest {
  readonly metadata: Omit<SkillVectorIndexMetadata, 'ready'>;
  readonly vectors: readonly SkillIndexVector[];
}

export interface SkillIndexWriteReceipt extends SkillVectorIndexMetadata {
  readonly writtenVectors: number;
}

/** Build-time mutation is intentionally separate from runtime search. */
export interface SkillIndexWriter {
  readonly capabilities: SkillAdapterCapabilities;
  replace(request: SkillIndexWriteRequest): Promise<SkillIndexWriteReceipt>;
  upsert?(request: SkillIndexWriteRequest): Promise<SkillIndexWriteReceipt>;
}

const MEMORY_CATALOG_CAPABILITIES: SkillAdapterCapabilities = {
  namespaces: true,
  lazyBodies: true,
  vectorSearch: false,
  indexWriting: false,
  eventuallyConsistent: false,
};

const MEMORY_VECTOR_CAPABILITIES: SkillAdapterCapabilities = {
  namespaces: true,
  lazyBodies: false,
  vectorSearch: true,
  indexWriting: true,
  eventuallyConsistent: false,
};

export interface InMemorySkillCatalogEntry {
  readonly descriptor: SkillDescriptor;
  readonly skill: AgentSkill;
}

/** Deterministic non-filesystem catalog used by tests and bundled runtimes. */
export class InMemorySkillCatalogStore implements SkillCatalogStore {
  readonly capabilities = MEMORY_CATALOG_CAPABILITIES;
  private readonly entries: readonly InMemorySkillCatalogEntry[];
  private readonly versions: ReadonlyMap<string, string>;

  constructor(
    entries: readonly InMemorySkillCatalogEntry[],
    versions: Readonly<Record<string, string>> = {},
  ) {
    this.entries = entries
      .slice()
      .sort((left, right) =>
        descriptorKey(left.descriptor).localeCompare(descriptorKey(right.descriptor)),
      );
    this.versions = new Map(Object.entries(versions));
  }

  async list(options: SkillCatalogListOptions = {}): Promise<readonly SkillDescriptor[]> {
    return this.entries
      .filter(({ descriptor }) => sameNamespace(descriptor.namespace, options.namespace))
      .map(({ descriptor }) => descriptor);
  }

  async load(name: string, options: SkillCatalogLoadOptions = {}): Promise<AgentSkill | undefined> {
    const entry = this.entries.find(
      ({ descriptor }) =>
        descriptor.name === name && sameNamespace(descriptor.namespace, options.namespace),
    );
    if (!entry) return undefined;
    if (
      options.expectedVersion != null &&
      (entry.descriptor.version ?? entry.descriptor.sourceHash) !== options.expectedVersion
    ) {
      throw new SkillAdapterError('STALE_CATALOG', `Skill '${name}' changed before activation`);
    }
    return entry.skill;
  }

  version(options: SkillCatalogListOptions = {}): Promise<string> {
    const key = namespaceKey(options.namespace);
    const configured = this.versions.get(key);
    return configured != null
      ? Promise.resolve(configured)
      : this.list(options).then(stableCatalogVersion);
  }

  health(options: SkillCatalogListOptions = {}): Promise<SkillAdapterHealth> {
    return this.version(options).then((checkedVersion) => ({ status: 'ready', checkedVersion }));
  }
}

interface StoredVector extends SkillIndexVector {
  readonly embedding: Float32Array;
}

interface MemoryVectorPartition {
  metadata: SkillVectorIndexMetadata;
  vectors: StoredVector[];
}

/** Exact cosine reference search with deterministic ordering and readiness checks. */
export class InMemorySkillVectorSearch implements SkillVectorSearch, SkillIndexWriter {
  readonly capabilities = MEMORY_VECTOR_CAPABILITIES;
  private readonly partitions = new Map<string, MemoryVectorPartition>();

  constructor(initial?: SkillIndexWriteRequest | readonly SkillIndexWriteRequest[]) {
    for (const request of initial == null ? [] : Array.isArray(initial) ? initial : [initial]) {
      this.store(request);
    }
  }

  metadata(options: SkillCatalogListOptions = {}): Promise<SkillVectorIndexMetadata> {
    return Promise.resolve(this.partition(options.namespace).metadata);
  }

  health(options: SkillCatalogListOptions = {}): Promise<SkillAdapterHealth> {
    const partition = this.partitions.get(namespaceKey(options.namespace));
    if (!partition?.metadata.ready) {
      return Promise.resolve({ status: 'not-ready', message: 'Index is not ready' });
    }
    return Promise.resolve({ status: 'ready', checkedVersion: partition.metadata.indexVersion });
  }

  async query(
    vector: ArrayLike<number>,
    options: SkillVectorQueryOptions = {},
  ): Promise<readonly SkillChunkMatch[]> {
    const partition = this.partition(options.namespace);
    assertReadyMetadata(partition.metadata, options);
    if (vector.length !== partition.metadata.dimensions) {
      throw new SkillAdapterError(
        'INVALID_RESPONSE',
        `Query embedding has ${vector.length} dimensions; expected ${partition.metadata.dimensions}`,
      );
    }
    const limit = options.limit ?? partition.vectors.length;
    if (!Number.isInteger(limit) || limit < 1) {
      throw new SkillAdapterError('INVALID_RESPONSE', 'limit must be a positive integer');
    }
    const minScore = options.minScore ?? Number.NEGATIVE_INFINITY;
    if (Number.isNaN(minScore)) {
      throw new SkillAdapterError('INVALID_RESPONSE', 'minScore must be a number');
    }
    return partition.vectors
      .map((entry) => ({
        name: entry.name,
        description: entry.description,
        chunk: entry.chunk,
        source: entry.source,
        score: cosine(vector, entry.embedding),
      }))
      .filter((match) => match.score >= minScore)
      .sort(
        (left, right) =>
          right.score - left.score ||
          left.name.localeCompare(right.name) ||
          left.chunk - right.chunk,
      )
      .slice(0, limit);
  }

  replace(request: SkillIndexWriteRequest): Promise<SkillIndexWriteReceipt> {
    return Promise.resolve(this.store(request));
  }

  upsert(request: SkillIndexWriteRequest): Promise<SkillIndexWriteReceipt> {
    const key = namespaceKey(request.metadata.namespace);
    const current = this.partitions.get(key);
    const merged = new Map<string, SkillIndexVector>();
    for (const vector of [...(current?.vectors ?? []), ...request.vectors]) {
      merged.set(`${vector.name}\0${vector.chunk}`, vector);
    }
    return Promise.resolve(this.store({ ...request, vectors: [...merged.values()] }));
  }

  private store(request: SkillIndexWriteRequest): SkillIndexWriteReceipt {
    validateWrite(request);
    const vectors = request.vectors.map((vector) => ({
      ...vector,
      embedding: Float32Array.from(vector.embedding),
    }));
    const metadata: SkillVectorIndexMetadata = { ...request.metadata, ready: true };
    this.partitions.set(namespaceKey(metadata.namespace), { metadata, vectors });
    return { ...metadata, writtenVectors: vectors.length };
  }

  private partition(namespace?: string): MemoryVectorPartition {
    const partition = this.partitions.get(namespaceKey(namespace));
    if (!partition) {
      throw new SkillAdapterError(
        'NOT_READY',
        `No ready skill index for namespace '${namespace ?? ''}'`,
      );
    }
    return partition;
  }
}

export function assertReadyMetadata(
  metadata: SkillVectorIndexMetadata,
  expected: Pick<
    SkillVectorQueryOptions,
    'catalogVersion' | 'indexVersion' | 'model' | 'revision' | 'embedderId'
  >,
): void {
  if (!metadata.ready) throw new SkillAdapterError('NOT_READY', 'Skill index is not ready');
  if (expected.catalogVersion != null && metadata.catalogVersion !== expected.catalogVersion) {
    throw new SkillAdapterError('STALE_CATALOG', 'Skill index catalog version is stale');
  }
  if (expected.indexVersion != null && metadata.indexVersion !== expected.indexVersion) {
    throw new SkillAdapterError('NOT_READY', 'Requested skill index version is not ready');
  }
  if (
    (expected.model != null && metadata.model !== expected.model) ||
    (expected.revision != null && metadata.revision !== expected.revision) ||
    (expected.embedderId != null && metadata.embedderId !== expected.embedderId)
  ) {
    throw new SkillAdapterError('MODEL_MISMATCH', 'Skill index and query embedder do not match');
  }
}

export function validateWrite(request: SkillIndexWriteRequest): void {
  if (!request.metadata.indexVersion || !request.metadata.catalogVersion) {
    throw new SkillAdapterError('INVALID_RESPONSE', 'Index and catalog versions are required');
  }
  if (!Number.isInteger(request.metadata.dimensions) || request.metadata.dimensions < 1) {
    throw new SkillAdapterError('INVALID_RESPONSE', 'Index dimensions must be a positive integer');
  }
  for (const vector of request.vectors) {
    if (vector.embedding.length !== request.metadata.dimensions) {
      throw new SkillAdapterError(
        'INVALID_RESPONSE',
        `Vector for '${vector.name}' has ${vector.embedding.length} dimensions; expected ${request.metadata.dimensions}`,
      );
    }
    for (let index = 0; index < vector.embedding.length; index++) {
      if (!Number.isFinite(Number(vector.embedding[index]))) {
        throw new SkillAdapterError('INVALID_RESPONSE', `Vector for '${vector.name}' is invalid`);
      }
    }
  }
}

function cosine(left: ArrayLike<number>, right: ArrayLike<number>): number {
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
  return leftNorm === 0 || rightNorm === 0 ? 0 : dot / Math.sqrt(leftNorm * rightNorm);
}

function descriptorKey(descriptor: SkillDescriptor): string {
  return `${namespaceKey(descriptor.namespace)}\0${descriptor.name}`;
}

function sameNamespace(left?: string, right?: string): boolean {
  return namespaceKey(left) === namespaceKey(right);
}

function namespaceKey(namespace?: string): string {
  return namespace ?? '';
}

function stableCatalogVersion(descriptors: readonly SkillDescriptor[]): string {
  const text = descriptors
    .map(
      (descriptor) => `${descriptor.name}\0${descriptor.sourceHash}\0${descriptor.version ?? ''}`,
    )
    .join('\n');
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index++) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return `fnv1a32:${(hash >>> 0).toString(16).padStart(8, '0')}`;
}
