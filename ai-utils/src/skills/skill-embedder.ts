export const DEFAULT_SKILL_EMBEDDING_MODEL = 'onnx-community/bge-small-en-v1.5-ONNX';

/**
 * Pinned model revision so an index can be reproduced and safely compared with
 * runtime query embeddings.
 */
export const DEFAULT_SKILL_EMBEDDING_REVISION = '5b661d92e0ec958f2ba968ec06819dd64d97c4c5';
export const DEFAULT_SKILL_EMBEDDING_DTYPE = 'q4';
export const DEFAULT_SKILL_EMBEDDING_POOLING = 'cls';
export const DEFAULT_SKILL_QUERY_PREFIX =
  'Represent this sentence for searching relevant passages: ';

/** Produces one normalized sentence embedding for each input string. */
export interface SkillEmbeddingOptions {
  readonly purpose?: 'document' | 'query';
}

export interface SkillTokenChunkOptions {
  /** Maximum tokenizer tokens per chunk, including {@link prefix}. */
  readonly maxTokens: number;
  /** Repeated content tokens between adjacent chunks. */
  readonly overlapTokens?: number;
  /** Text prepended to every chunk, such as the skill name. */
  readonly prefix?: string;
}

export interface SkillEmbedder {
  /** Stable identity for build/runtime compatibility checks. */
  readonly id: string;
  readonly model: string;
  readonly revision: string;
  embed(
    texts: readonly string[],
    options?: SkillEmbeddingOptions,
  ): Promise<readonly Float32Array[]>;
  /** Split text using the embedding model's own tokenizer. */
  split(text: string, options: SkillTokenChunkOptions): Promise<readonly string[]>;
}

export interface TransformersJsSkillEmbedderOptions {
  readonly model?: string;
  readonly revision?: string;
  readonly dtype?: string;
  readonly pooling?: 'mean' | 'cls';
  /** Retrieval models such as BGE recommend a prefix for queries only. */
  readonly queryPrefix?: string;
  /** Override module loading for alternate runtimes or deterministic tests. */
  readonly loadTransformers?: () => Promise<unknown>;
}

type FeatureExtractionOutput = {
  readonly data: ArrayLike<number>;
  readonly dims: readonly number[];
};

type FeatureExtractor = (
  texts: string | readonly string[],
  options: { pooling: 'mean' | 'cls'; normalize: true },
) => Promise<FeatureExtractionOutput>;

type TransformersTokenizer = {
  encode(text: string, options: { add_special_tokens: false }): number[];
  decode(
    tokenIds: number[],
    options: { skip_special_tokens: true; clean_up_tokenization_spaces: false },
  ): string;
};

type TransformersJsModule = {
  AutoTokenizer: {
    from_pretrained(model: string, options: { revision: string }): Promise<TransformersTokenizer>;
  };
  AutoModel: {
    from_pretrained(model: string, options: { revision: string; dtype: string }): Promise<unknown>;
  };
  FeatureExtractionPipeline: new (options: {
    task: 'feature-extraction';
    tokenizer: unknown;
    model: unknown;
  }) => FeatureExtractor;
};

/**
 * Local sentence embeddings powered by Transformers.js. The package and model
 * are loaded lazily, so small skill catalogs never pay their startup cost.
 */
export class TransformersJsSkillEmbedder implements SkillEmbedder {
  readonly id: string;
  readonly model: string;
  readonly revision: string;
  readonly dtype: string;
  readonly pooling: 'mean' | 'cls';
  readonly queryPrefix: string;
  private tokenizerPromise?: Promise<TransformersTokenizer>;
  private extractorPromise?: Promise<FeatureExtractor>;
  private readonly transformersImporter?: () => Promise<unknown>;

  constructor(options: TransformersJsSkillEmbedderOptions = {}) {
    this.model = options.model ?? DEFAULT_SKILL_EMBEDDING_MODEL;
    this.revision = options.revision ?? DEFAULT_SKILL_EMBEDDING_REVISION;
    this.dtype = options.dtype ?? DEFAULT_SKILL_EMBEDDING_DTYPE;
    const usesDefaultModel = this.model === DEFAULT_SKILL_EMBEDDING_MODEL;
    this.pooling = options.pooling ?? (usesDefaultModel ? DEFAULT_SKILL_EMBEDDING_POOLING : 'mean');
    this.queryPrefix = options.queryPrefix ?? (usesDefaultModel ? DEFAULT_SKILL_QUERY_PREFIX : '');
    this.transformersImporter = options.loadTransformers;
    this.id = `transformers.js@4.2.0:${this.model}@${this.revision}:dtype=${this.dtype}:pooling=${this.pooling}:query-prefix=${JSON.stringify(this.queryPrefix)}:l2`;
  }

  async embed(
    texts: readonly string[],
    options: SkillEmbeddingOptions = {},
  ): Promise<readonly Float32Array[]> {
    if (texts.length === 0) return [];

    const extractor = await this.extractor();
    const input =
      options.purpose === 'query' && this.queryPrefix
        ? texts.map((text) => `${this.queryPrefix}${text}`)
        : texts;
    const output = await extractor(input, { pooling: this.pooling, normalize: true });
    const batch = output.dims[0];
    const dimensions = output.dims[output.dims.length - 1];
    if (output.dims.length !== 2 || batch !== texts.length || dimensions == null) {
      throw new Error(
        `Unexpected Transformers.js embedding shape: [${output.dims.join(', ')}] for ${texts.length} inputs`,
      );
    }
    if (output.data.length !== batch * dimensions) {
      throw new Error(
        `Unexpected Transformers.js embedding length: ${output.data.length}; expected ${batch * dimensions}`,
      );
    }

    const vectors: Float32Array[] = [];
    for (let row = 0; row < batch; row++) {
      const vector = new Float32Array(dimensions);
      const offset = row * dimensions;
      for (let column = 0; column < dimensions; column++) {
        vector[column] = Number(output.data[offset + column]);
      }
      vectors.push(vector);
    }
    return vectors;
  }

  async split(text: string, options: SkillTokenChunkOptions): Promise<readonly string[]> {
    const maxTokens = positiveInteger(options.maxTokens, 'maxTokens');
    const overlapTokens = nonNegativeInteger(options.overlapTokens ?? 0, 'overlapTokens');
    const prefix = options.prefix ?? '';
    const tokenizer = await this.tokenizer();
    const prefixTokens = tokenizer.encode(prefix, { add_special_tokens: false });
    const contentLimit = maxTokens - prefixTokens.length;
    if (contentLimit < 1) {
      throw new Error(
        `Chunk prefix uses ${prefixTokens.length} tokens, leaving no room in maxTokens=${maxTokens}`,
      );
    }
    if (overlapTokens >= contentLimit) {
      throw new Error(
        `overlapTokens (${overlapTokens}) must be smaller than the ${contentLimit}-token content window`,
      );
    }

    const tokenIds = tokenizer.encode(text, { add_special_tokens: false });
    if (tokenIds.length === 0) return [prefix.trimEnd()];

    const chunks: string[] = [];
    for (let start = 0; start < tokenIds.length; start += contentLimit - overlapTokens) {
      const end = Math.min(start + contentLimit, tokenIds.length);
      const decoded = tokenizer.decode(tokenIds.slice(start, end), {
        skip_special_tokens: true,
        clean_up_tokenization_spaces: false,
      });
      chunks.push(`${prefix}${decoded}`.trim());
      if (end === tokenIds.length) break;
    }
    return chunks;
  }

  private tokenizer(): Promise<TransformersTokenizer> {
    this.tokenizerPromise ??= loadTransformersJs(this.transformersImporter).then((transformers) =>
      transformers.AutoTokenizer.from_pretrained(this.model, { revision: this.revision }),
    );
    return this.tokenizerPromise;
  }

  private extractor(): Promise<FeatureExtractor> {
    this.extractorPromise ??= loadTransformersJs(this.transformersImporter).then(
      async (transformers) => {
        // Construct the pipeline from pinned components. Transformers.js 4.2's
        // high-level pipeline performs an unpinned metadata probe before loading.
        const [tokenizer, model] = await Promise.all([
          this.tokenizer(),
          transformers.AutoModel.from_pretrained(this.model, {
            revision: this.revision,
            dtype: this.dtype,
          }),
        ]);
        return new transformers.FeatureExtractionPipeline({
          task: 'feature-extraction',
          tokenizer,
          model,
        });
      },
    );
    return this.extractorPromise;
  }
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative integer`);
  }
  return value;
}

export async function loadTransformersJs(
  importer: () => Promise<unknown> = async () => {
    // Keep this dependency out of the normal skill path and library bundle.
    const packageName = '@huggingface/transformers';
    return import(packageName);
  },
): Promise<TransformersJsModule> {
  try {
    return (await importer()) as TransformersJsModule;
  } catch (error) {
    const detail = error instanceof Error ? ` (${error.message})` : '';
    throw new Error(
      `The default semantic skill indexer requires the optional peer @huggingface/transformers. Install it with your package manager (for example, bun add @huggingface/transformers@4.2.0) or configure a custom SkillEmbedder${detail}`,
      { cause: error },
    );
  }
}
