import type { Document } from '../../document/document.ts';
import { withDocumentScore } from '../../document/document.ts';
import type { EmbeddingModel } from '../../embedding/embedding-model.ts';
import { cosineSimilarity } from '../../embedding/fake-embedding-model.ts';
import type { FilterExpression, FilterOperand } from '../filter/filter.ts';
import { resolveDocumentEmbedding, resolveQueryEmbedding } from '../resolve-embedding.ts';
import { type SearchRequest, searchRequest } from '../search-request.ts';
import type { VectorStore } from '../vector-store.ts';

/**
 * Record representing a single vector entry in S3 Vectors.
 */
export interface S3VectorRecord {
  readonly id: string;
  readonly vector?: readonly number[];
  readonly metadata?: Readonly<Record<string, unknown>>;
  readonly document?: string;
  readonly score?: number;
}

export interface S3VectorsPutInput {
  readonly vectorBucketName: string;
  readonly indexName: string;
  readonly vectors: readonly S3VectorRecord[];
}

export interface S3VectorsQueryInput {
  readonly vectorBucketName: string;
  readonly indexName: string;
  readonly queryVector: readonly number[];
  readonly topK?: number;
  readonly filter?: unknown;
  readonly returnMetadata?: boolean;
  readonly returnDocument?: boolean;
}

export interface S3VectorsQueryResult {
  readonly vectors?: readonly S3VectorRecord[];
}

export interface S3VectorsDeleteInput {
  readonly vectorBucketName: string;
  readonly indexName: string;
  readonly ids?: readonly string[];
  readonly filter?: unknown;
}

export interface S3VectorsGetInput {
  readonly vectorBucketName: string;
  readonly indexName: string;
  readonly id: string;
}

/**
 * Pluggable client contract for AWS S3 Vectors.
 */
export interface S3VectorsClient {
  putVectors(input: S3VectorsPutInput): Promise<void | unknown>;
  queryVectors(input: S3VectorsQueryInput): Promise<S3VectorsQueryResult>;
  deleteVectors(input: S3VectorsDeleteInput): Promise<void | unknown>;
  getVector?(input: S3VectorsGetInput): Promise<S3VectorRecord | null>;
}

/**
 * Configuration options for {@link S3VectorStore}.
 */
export interface S3VectorStoreOptions {
  /** AWS S3 Vector Bucket name */
  readonly vectorBucketName: string;
  /** S3 Vectors Index name (e.g. 'di-framework-ai-index') */
  readonly indexName: string;
  /** Embedding model instance used to compute document and query embeddings */
  readonly embeddingModel: EmbeddingModel;
  /** AWS Region */
  readonly region?: string;
  /** Optional S3 Vectors client instance */
  readonly client?: S3VectorsClient | unknown;
  /** Optional custom vector store name */
  readonly name?: string;
}

/**
 * Translates a portable {@link FilterExpression} AST into an S3 Vectors metadata filter dictionary.
 */
export function translateS3FilterExpression(
  expression: FilterExpression | null | undefined,
): Record<string, unknown> | null {
  if (!expression) return null;

  function translateOperand(operand: FilterOperand): unknown {
    if (operand.kind === 'key') {
      let k = operand.key;
      if (
        k.length >= 2 &&
        ((k.startsWith('"') && k.endsWith('"')) || (k.startsWith("'") && k.endsWith("'")))
      ) {
        k = k.slice(1, -1);
      }
      return k;
    }
    if (operand.kind === 'value') {
      return operand.value;
    }
    if (operand.kind === 'group') {
      return translateNode(operand.content);
    }
    if (operand.kind === 'expression') {
      return translateNode(operand);
    }
    return null;
  }

  function translateNode(expr: FilterExpression): Record<string, unknown> {
    switch (expr.type) {
      case 'AND': {
        const left = translateOperand(expr.left);
        const right = expr.right ? translateOperand(expr.right) : null;
        return { $and: [left, right].filter(Boolean) };
      }
      case 'OR': {
        const left = translateOperand(expr.left);
        const right = expr.right ? translateOperand(expr.right) : null;
        return { $or: [left, right].filter(Boolean) };
      }
      case 'NOT': {
        const left = translateOperand(expr.left);
        return { $not: left };
      }
      case 'EQ': {
        const key = String(translateOperand(expr.left));
        const val = expr.right ? translateOperand(expr.right) : null;
        return { [key]: { $eq: val } };
      }
      case 'NE': {
        const key = String(translateOperand(expr.left));
        const val = expr.right ? translateOperand(expr.right) : null;
        return { [key]: { $ne: val } };
      }
      case 'GT': {
        const key = String(translateOperand(expr.left));
        const val = expr.right ? translateOperand(expr.right) : null;
        return { [key]: { $gt: val } };
      }
      case 'GTE': {
        const key = String(translateOperand(expr.left));
        const val = expr.right ? translateOperand(expr.right) : null;
        return { [key]: { $gte: val } };
      }
      case 'LT': {
        const key = String(translateOperand(expr.left));
        const val = expr.right ? translateOperand(expr.right) : null;
        return { [key]: { $lt: val } };
      }
      case 'LTE': {
        const key = String(translateOperand(expr.left));
        const val = expr.right ? translateOperand(expr.right) : null;
        return { [key]: { $lte: val } };
      }
      case 'IN': {
        const key = String(translateOperand(expr.left));
        const val = expr.right ? translateOperand(expr.right) : null;
        const list = Array.isArray(val) ? val : [val];
        return { [key]: { $in: list } };
      }
      case 'NIN': {
        const key = String(translateOperand(expr.left));
        const val = expr.right ? translateOperand(expr.right) : null;
        const list = Array.isArray(val) ? val : [val];
        return { [key]: { $nin: list } };
      }
      case 'ISNULL': {
        const key = String(translateOperand(expr.left));
        return { [key]: { $exists: false } };
      }
      case 'ISNOTNULL': {
        const key = String(translateOperand(expr.left));
        return { [key]: { $exists: true } };
      }
      default:
        throw new Error(
          `Unsupported S3 filter expression type: ${(expr as { type: string }).type}`,
        );
    }
  }

  return translateNode(expression);
}

export { cosineSimilarity };

function compareValues(a: unknown, b: unknown): number {
  if (a == null && b == null) return 0;
  if (a == null) return -1;
  if (b == null) return 1;
  if (typeof a === 'number' && typeof b === 'number') {
    return a === b ? 0 : a < b ? -1 : 1;
  }
  if (typeof a === 'boolean' && typeof b === 'boolean') {
    return a === b ? 0 : a ? 1 : -1;
  }
  const strA = String(a);
  const strB = String(b);
  return strA === strB ? 0 : strA < strB ? -1 : 1;
}

/**
 * Evaluates whether metadata matches a translated S3 Vectors filter dictionary.
 */
export function matchesS3Filter(
  metadata: Readonly<Record<string, unknown>>,
  filter: unknown,
): boolean {
  if (!filter || typeof filter !== 'object') return true;

  const f = filter as Record<string, unknown>;

  if ('$and' in f && Array.isArray(f.$and)) {
    return f.$and.every((sub) => matchesS3Filter(metadata, sub));
  }
  if ('$or' in f && Array.isArray(f.$or)) {
    return f.$or.some((sub) => matchesS3Filter(metadata, sub));
  }
  if ('$not' in f) {
    return !matchesS3Filter(metadata, f.$not);
  }

  for (const [key, condition] of Object.entries(f)) {
    const actual = metadata[key];
    if (condition && typeof condition === 'object') {
      const cond = condition as Record<string, unknown>;
      if ('$eq' in cond) {
        if (compareValues(actual, cond.$eq) !== 0) return false;
      }
      if ('$ne' in cond) {
        if (compareValues(actual, cond.$ne) === 0) return false;
      }
      if ('$gt' in cond) {
        if (compareValues(actual, cond.$gt) <= 0) return false;
      }
      if ('$gte' in cond) {
        if (compareValues(actual, cond.$gte) < 0) return false;
      }
      if ('$lt' in cond) {
        if (compareValues(actual, cond.$lt) >= 0) return false;
      }
      if ('$lte' in cond) {
        if (compareValues(actual, cond.$lte) > 0) return false;
      }
      if ('$in' in cond && Array.isArray(cond.$in)) {
        if (!cond.$in.some((item) => compareValues(actual, item) === 0)) return false;
      }
      if ('$nin' in cond && Array.isArray(cond.$nin)) {
        if (cond.$nin.some((item) => compareValues(actual, item) === 0)) return false;
      }
      if ('$exists' in cond) {
        const exists = actual !== undefined && actual !== null;
        if (Boolean(cond.$exists) !== exists) return false;
      }
    } else {
      if (compareValues(actual, condition) !== 0) return false;
    }
  }

  return true;
}

/**
 * In-memory S3 Vectors client for testing and local vector execution.
 */
export class InMemoryS3VectorsClient implements S3VectorsClient {
  private readonly records = new Map<string, S3VectorRecord>();

  async putVectors(input: S3VectorsPutInput): Promise<void> {
    for (const v of input.vectors) {
      this.records.set(v.id, v);
    }
  }

  async getVector(input: S3VectorsGetInput): Promise<S3VectorRecord | null> {
    return this.records.get(input.id) ?? null;
  }

  async deleteVectors(input: S3VectorsDeleteInput): Promise<void> {
    if (input.ids) {
      for (const id of input.ids) {
        this.records.delete(id);
      }
    }
    if (input.filter) {
      for (const [id, rec] of this.records.entries()) {
        if (matchesS3Filter(rec.metadata ?? {}, input.filter)) {
          this.records.delete(id);
        }
      }
    }
  }

  async queryVectors(input: S3VectorsQueryInput): Promise<S3VectorsQueryResult> {
    const scored: Array<S3VectorRecord & { score: number }> = [];
    for (const rec of this.records.values()) {
      if (input.filter && !matchesS3Filter(rec.metadata ?? {}, input.filter)) {
        continue;
      }
      const score = rec.vector ? cosineSimilarity(rec.vector, input.queryVector) : 0;
      scored.push({ ...rec, score });
    }
    scored.sort((a, b) => b.score - a.score);
    const top = input.topK ? scored.slice(0, input.topK) : scored;
    return {
      vectors: top.map((r) => ({
        id: r.id,
        score: r.score,
        document: input.returnDocument !== false ? r.document : undefined,
        metadata: input.returnMetadata !== false ? r.metadata : undefined,
        vector: r.vector,
      })),
    };
  }
}

/**
 * AWS S3 Vector Store adapter for high-scale, serverless vector search.
 * Spring AI: {@code S3VectorStore}.
 */
export class S3VectorStore implements VectorStore {
  readonly name: string;
  readonly vectorBucketName: string;
  readonly indexName: string;
  readonly region?: string;
  private readonly client: S3VectorsClient;
  private readonly model: EmbeddingModel;
  private readonly docs = new Map<string, Document>();

  constructor(options: S3VectorStoreOptions) {
    this.vectorBucketName = options.vectorBucketName;
    this.indexName = options.indexName;
    this.region = options.region;
    this.model = options.embeddingModel;
    this.name = options.name ?? 'S3VectorStore';
    this.client = (options.client as S3VectorsClient) ?? new InMemoryS3VectorsClient();
  }

  /**
   * Embeds documents and upserts vectors and metadata into the S3 Vector index.
   */
  async add(documents: readonly Document[]): Promise<void> {
    const vectors: S3VectorRecord[] = [];
    for (const doc of documents) {
      const values = await resolveDocumentEmbedding(this.model, doc);
      const text = doc.text ?? '';
      vectors.push({
        id: doc.id,
        vector: values,
        document: text,
        metadata: { ...doc.metadata, text },
      });
      this.docs.set(doc.id, doc);
    }
    await this.client.putVectors({
      vectorBucketName: this.vectorBucketName,
      indexName: this.indexName,
      vectors,
    });
  }

  /**
   * Retrieves a document by its unique ID.
   */
  async get(id: string): Promise<Document | null> {
    if (this.client.getVector) {
      const rec = await this.client.getVector({
        vectorBucketName: this.vectorBucketName,
        indexName: this.indexName,
        id,
      });
      if (!rec) return null;
      return {
        id: rec.id,
        text: rec.document ?? (typeof rec.metadata?.text === 'string' ? rec.metadata.text : ''),
        media: null,
        metadata: rec.metadata ?? {},
        score: null,
      };
    }
    return this.docs.get(id) ?? null;
  }

  /**
   * Deletes vectors by IDs from the S3 Vector index.
   */
  async delete(ids: readonly string[]): Promise<void> {
    await this.client.deleteVectors({
      vectorBucketName: this.vectorBucketName,
      indexName: this.indexName,
      ids: [...ids],
    });
    for (const id of ids) {
      this.docs.delete(id);
    }
  }

  /**
   * Deletes vectors matching a portable metadata filter expression.
   */
  async deleteByFilter(filterExpression: FilterExpression): Promise<void> {
    const filter = translateS3FilterExpression(filterExpression);
    await this.client.deleteVectors({
      vectorBucketName: this.vectorBucketName,
      indexName: this.indexName,
      filter,
    });
    for (const [id, doc] of this.docs.entries()) {
      if (matchesS3Filter(doc.metadata ?? {}, filter)) {
        this.docs.delete(id);
      }
    }
  }

  /**
   * Executes similarity search with topK, similarityThreshold, and metadata filtering.
   */
  async similaritySearch(request: SearchRequest): Promise<readonly Document[]> {
    const queryVector = await resolveQueryEmbedding(this.model, request);
    const filter = request.filterExpression
      ? translateS3FilterExpression(request.filterExpression)
      : undefined;

    const result = await this.client.queryVectors({
      vectorBucketName: this.vectorBucketName,
      indexName: this.indexName,
      queryVector,
      topK: request.topK,
      filter,
      returnMetadata: true,
      returnDocument: true,
    });

    const matches = result?.vectors ?? [];
    return matches
      .filter((m) => (m.score ?? 0) >= request.similarityThreshold)
      .map((m) => {
        const cached = this.docs.get(m.id);
        const text =
          m.document ??
          (typeof m.metadata?.text === 'string' ? m.metadata.text : (cached?.text ?? ''));
        const doc: Document = cached
          ? {
              ...cached,
              text: m.document ?? cached.text,
              metadata: m.metadata ?? cached.metadata,
            }
          : {
              id: m.id,
              text,
              media: null,
              metadata: m.metadata ?? {},
              score: null,
            };
        return withDocumentScore(doc, m.score ?? 0);
      });
  }

  /**
   * Convenience similarity search from text query.
   */
  similaritySearchQuery(query: string): Promise<readonly Document[]> {
    return this.similaritySearch(searchRequest({ query }));
  }
}
