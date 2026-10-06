import type { Document } from '../../document/document.ts';
import type { EmbeddingModel } from '../../embedding/embedding-model.ts';
import { AiError, isAiError } from '../../model/errors.ts';
import { resolveWorkersAiBinding, type WorkersAiBindingSource } from './binding.ts';

export const DEFAULT_WORKERS_AI_EMBEDDING_MODEL = '@cf/baai/bge-base-en-v1.5';

export interface WorkersAiEmbeddingOptions {
  readonly binding: WorkersAiBindingSource;
  readonly model?: string;
  readonly dimensions?: number;
}

/**
 * Embeds text with a Cloudflare Workers AI binding.
 * The binding is resolved on each call, including `{ binding }` descriptors
 * from `@di-framework/cloudflare`.
 */
export class WorkersAiEmbeddingModel implements EmbeddingModel {
  readonly dimensions?: number;
  private readonly options: WorkersAiEmbeddingOptions;

  constructor(options: WorkersAiEmbeddingOptions) {
    this.options = options;
    if (options.dimensions !== undefined) this.dimensions = options.dimensions;
  }

  static of(
    binding: WorkersAiBindingSource,
    options: Omit<WorkersAiEmbeddingOptions, 'binding'> = {},
  ): WorkersAiEmbeddingModel {
    return new WorkersAiEmbeddingModel({ ...options, binding });
  }

  embed(text: string): Promise<number[]> {
    return this.embedBatch([text]).then((vectors) => vectors[0] ?? []);
  }

  embedDocument(document: Document): Promise<number[]> {
    return this.embed(document.text ?? '');
  }

  async embedBatch(texts: readonly string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const model = this.options.model ?? DEFAULT_WORKERS_AI_EMBEDDING_MODEL;
    const binding = resolveWorkersAiBinding(this.options.binding);
    let raw: unknown;
    try {
      raw = await binding.run(model, { text: [...texts] });
    } catch (error) {
      if (isAiError(error)) throw error;
      const message = error instanceof Error ? error.message : 'Workers AI embedding failed';
      throw new AiError(message, 'provider-error', { provider: 'workers-ai', model, cause: error });
    }
    return vectorsFrom(raw, texts.length, model);
  }
}

export function workersAiEmbeddingModel(
  options: WorkersAiEmbeddingOptions,
): WorkersAiEmbeddingModel {
  return new WorkersAiEmbeddingModel(options);
}

function vectorsFrom(raw: unknown, count: number, model: string): number[][] {
  const record = unwrap(raw);
  const data = record.data;
  if (Array.isArray(data) && data.every((row) => Array.isArray(row))) {
    return data.map((row) => row.map((value) => Number(value)));
  }
  if (Array.isArray(data) && data.every((value) => typeof value === 'number')) {
    const shape = record.shape;
    const dimensions = Array.isArray(shape) && typeof shape[1] === 'number' ? shape[1] : undefined;
    if (dimensions && dimensions > 0 && data.length >= dimensions) {
      const rows: number[][] = [];
      for (let index = 0; index < data.length; index += dimensions) {
        rows.push(data.slice(index, index + dimensions) as number[]);
      }
      return rows;
    }
    if (count === 1) return [data as number[]];
  }
  throw new AiError('Workers AI returned no embeddings', 'provider-error', {
    provider: 'workers-ai',
    model,
    retryable: false,
  });
}

function unwrap(raw: unknown): Record<string, unknown> {
  if (!isRecord(raw)) return {};
  if (isRecord(raw.result)) return raw.result;
  return raw;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
