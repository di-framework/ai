import { describe, expect, test } from 'bun:test';
import {
  DEFAULT_SKILL_EMBEDDING_DTYPE,
  DEFAULT_SKILL_EMBEDDING_MODEL,
  DEFAULT_SKILL_EMBEDDING_POOLING,
  DEFAULT_SKILL_EMBEDDING_REVISION,
  DEFAULT_SKILL_QUERY_PREFIX,
  TransformersJsSkillEmbedder,
} from '../src/index.ts';
import { loadTransformersJs } from '../src/skills/skill-embedder.ts';

type FakeOutput = { data: ArrayLike<number>; dims: readonly number[] };

function fakeTransformers(output: FakeOutput | ((texts: readonly string[]) => FakeOutput)) {
  const calls: {
    tokenizer: unknown[];
    model: unknown[];
    pipeline: unknown[];
    inputs: Array<readonly string[]>;
    options: unknown[];
  } = { tokenizer: [], model: [], pipeline: [], inputs: [], options: [] };
  const tokenizer = {
    encode(text: string) {
      return [...text].map((character) => character.codePointAt(0) ?? 0);
    },
    decode(ids: number[]) {
      return String.fromCodePoint(...ids);
    },
  };
  function FeatureExtractionPipeline(options: unknown) {
    calls.pipeline.push(options);
    return async (input: string | readonly string[], extractionOptions: unknown) => {
      const texts = typeof input === 'string' ? [input] : [...input];
      calls.inputs.push(texts);
      calls.options.push(extractionOptions);
      return typeof output === 'function' ? output(texts) : output;
    };
  }
  const module = {
    AutoTokenizer: {
      async from_pretrained(...args: unknown[]) {
        calls.tokenizer.push(args);
        return tokenizer;
      },
    },
    AutoModel: {
      async from_pretrained(...args: unknown[]) {
        calls.model.push(args);
        return { fake: 'model' };
      },
    },
    FeatureExtractionPipeline,
  };
  return { calls, module };
}

describe('TransformersJsSkillEmbedder', () => {
  test('uses pinned defaults and keeps empty embedding batches lazy', async () => {
    const embedder = new TransformersJsSkillEmbedder();
    expect(embedder).toMatchObject({
      model: DEFAULT_SKILL_EMBEDDING_MODEL,
      revision: DEFAULT_SKILL_EMBEDDING_REVISION,
      dtype: DEFAULT_SKILL_EMBEDDING_DTYPE,
      pooling: DEFAULT_SKILL_EMBEDDING_POOLING,
      queryPrefix: DEFAULT_SKILL_QUERY_PREFIX,
    });
    expect(embedder.id).toContain('transformers.js@4.2.0');
    await expect(embedder.embed([])).resolves.toEqual([]);
  });

  test('loads pinned components once, prefixes queries, and returns row vectors', async () => {
    const fake = fakeTransformers((texts) => ({
      dims: [texts.length, 2],
      data: texts.flatMap((_, index) => [index + 1, index + 2]),
    }));
    const embedder = new TransformersJsSkillEmbedder({
      model: 'custom-model',
      revision: 'custom-revision',
      dtype: 'fp32',
      pooling: 'cls',
      queryPrefix: 'query: ',
      loadTransformers: async () => fake.module,
    });

    expect(await embedder.embed(['one', 'two'], { purpose: 'query' })).toEqual([
      new Float32Array([1, 2]),
      new Float32Array([2, 3]),
    ]);
    expect(await embedder.embed(['document'])).toEqual([new Float32Array([1, 2])]);
    expect(fake.calls.inputs).toEqual([['query: one', 'query: two'], ['document']]);
    expect(fake.calls.options[0]).toEqual({ pooling: 'cls', normalize: true });
    expect(fake.calls.tokenizer).toHaveLength(1);
    expect(fake.calls.model).toHaveLength(1);
    expect(fake.calls.pipeline).toHaveLength(1);
  });

  test('defaults custom models to mean pooling without a query prefix', () => {
    expect(new TransformersJsSkillEmbedder({ model: 'another-model' })).toMatchObject({
      pooling: 'mean',
      queryPrefix: '',
    });
  });

  test('rejects malformed embedding shapes and lengths', async () => {
    const badShape = fakeTransformers({ dims: [1, 1, 2], data: [1, 2] });
    await expect(
      new TransformersJsSkillEmbedder({
        loadTransformers: async () => badShape.module,
      }).embed(['one']),
    ).rejects.toThrow(/embedding shape/);

    const badLength = fakeTransformers({ dims: [1, 2], data: [1] });
    await expect(
      new TransformersJsSkillEmbedder({
        loadTransformers: async () => badLength.module,
      }).embed(['one']),
    ).rejects.toThrow(/embedding length/);
  });

  test('token-chunks with overlap and handles empty text and prefixes', async () => {
    const fake = fakeTransformers({ dims: [1, 1], data: [1] });
    const embedder = new TransformersJsSkillEmbedder({
      loadTransformers: async () => fake.module,
    });
    expect(await embedder.split('abcdef', { maxTokens: 4, overlapTokens: 1 })).toEqual([
      'abcd',
      'def',
    ]);
    expect(await embedder.split('', { maxTokens: 10, prefix: 'route: ' })).toEqual(['route:']);
    expect(await embedder.split('abcd', { maxTokens: 5, prefix: 'x' })).toEqual(['xabcd']);
  });

  test('validates chunk token windows', async () => {
    const fake = fakeTransformers({ dims: [1, 1], data: [1] });
    const embedder = new TransformersJsSkillEmbedder({
      loadTransformers: async () => fake.module,
    });
    await expect(embedder.split('x', { maxTokens: 0 })).rejects.toThrow(/positive integer/);
    await expect(embedder.split('x', { maxTokens: 2, overlapTokens: -1 })).rejects.toThrow(
      /non-negative integer/,
    );
    await expect(embedder.split('x', { maxTokens: 2, prefix: 'xx' })).rejects.toThrow(
      /leaving no room/,
    );
    await expect(
      embedder.split('x', { maxTokens: 3, prefix: 'x', overlapTokens: 2 }),
    ).rejects.toThrow(/must be smaller/);
  });

  test('loads the optional module and wraps loader failures with guidance', async () => {
    expect((await loadTransformersJs()).AutoTokenizer).toBeDefined();
    await expect(
      loadTransformersJs(async () => {
        throw new Error('missing');
      }),
    ).rejects.toThrow(/optional peer.*missing/);
    await expect(loadTransformersJs(async () => Promise.reject('missing'))).rejects.toThrow(
      /optional peer/,
    );
  });
});
