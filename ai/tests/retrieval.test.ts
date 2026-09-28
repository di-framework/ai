import { describe, expect, test } from 'bun:test';
import type { ChatClientResponse } from '../src/chat/client/chat-client-response.ts';
import {
  ChatClient,
  ConcatenationDocumentJoiner,
  ContextualQueryAugmenter,
  cosineSimilarity,
  document,
  evaluateFilterExpression,
  FakeChatModel,
  FakeEmbeddingModel,
  FilterExpressionBuilder,
  filterExpression,
  filterKey,
  filterValue,
  isFilterExpression,
  PrecomputedEmbeddingModel,
  parseFilterExpression,
  query,
  RAG_DOCUMENT_CONTEXT,
  RecordingChatModel,
  RetrievalAugmentationAdvisor,
  SimpleVectorStore,
  searchRequest,
  textDocument,
  VectorStoreDocumentRetriever,
} from '../src/index.ts';
import { mutateQuery } from '../src/rag/query.ts';

describe('ConcatenationDocumentJoiner', () => {
  test('concatenates and de-duplicates documents by id (first wins)', () => {
    const joiner = new ConcatenationDocumentJoiner();
    const d1 = textDocument('doc one', {}, 'd1');
    const d1Dup = textDocument('doc one duplicate', {}, 'd1');
    const d2 = textDocument('doc two', {}, 'd2');
    const q1 = query('q1');
    const q2 = query('q2');

    const result = joiner.join(
      new Map([
        [q1, [[d1], [d2]]],
        [q2, [[d1Dup]]],
      ]),
    );

    expect(result).toHaveLength(2);
    expect(result[0]).toBe(d1);
    expect(result.map((d) => d.id)).toEqual(['d1', 'd2']);
  });
});

describe('query / mutateQuery', () => {
  test('object form defaults history/context and throws when text is missing', () => {
    const q = query({ text: 'hi' });
    expect(q).toEqual({ text: 'hi', history: [], context: {} });
    expect(() => query({ text: null as never })).toThrow('text cannot be null');
  });

  test('mutateQuery overrides only the provided fields', () => {
    const original = query({ text: 'original', context: { a: 1 } });
    const mutated = mutateQuery(original, { text: 'changed' });
    expect(mutated).toEqual({ text: 'changed', history: [], context: { a: 1 } });
  });
});

describe('SearchRequest', () => {
  test('requires a positive topK', () => {
    expect(() => searchRequest({ topK: 0 })).toThrow('TopK should be positive.');
    expect(() => searchRequest({ topK: -1 })).toThrow('TopK should be positive.');
    expect(searchRequest({ topK: 1 }).topK).toBe(1);
  });
});

describe('FakeEmbeddingModel / cosine', () => {
  test('similar texts score higher than unrelated', () => {
    const model = new FakeEmbeddingModel({ dimensions: 64 });
    const a = model.embed('Yorktown Virginia historic town');
    const b = model.embed('Yorktown is a town in Virginia');
    const c = model.embed('quantum chromodynamics lattice');
    expect(cosineSimilarity(a, b)).toBeGreaterThan(cosineSimilarity(a, c));
  });

  test('embedBatch embeds each text independently', () => {
    const model = new FakeEmbeddingModel({ dimensions: 16 });
    const [a, b] = model.embedBatch(['hello world', 'goodbye world']);
    expect(a).toHaveLength(16);
    expect(b).toHaveLength(16);
    expect(a).not.toEqual(b);
    expect(model.embed('hello world')).toEqual(a!);
  });
});

describe('Filter expressions', () => {
  const b = new FilterExpressionBuilder();

  test('builder EQ and AND', () => {
    const exp = b.and(b.eq('country', 'UK'), b.gte('year', 2020)).build();
    expect(evaluateFilterExpression(exp, { country: 'UK', year: 2021 })).toBe(true);
    expect(evaluateFilterExpression(exp, { country: 'UK', year: 2019 })).toBe(false);
    expect(evaluateFilterExpression(exp, { country: 'US', year: 2021 })).toBe(false);
  });

  test('IN / NIN', () => {
    const inExp = b.in('genre', 'comedy', 'drama').build();
    expect(evaluateFilterExpression(inExp, { genre: 'drama' })).toBe(true);
    expect(evaluateFilterExpression(inExp, { genre: 'horror' })).toBe(false);

    const ninExp = b.nin('city', 'Sofia', 'Plovdiv').build();
    expect(evaluateFilterExpression(ninExp, { city: 'Varna' })).toBe(true);
    expect(evaluateFilterExpression(ninExp, { city: 'Sofia' })).toBe(false);
  });

  test('NE / GT / LT / OR / group', () => {
    expect(evaluateFilterExpression(b.ne('country', 'UK').build(), { country: 'US' })).toBe(true);
    expect(evaluateFilterExpression(b.ne('country', 'UK').build(), { country: 'UK' })).toBe(false);
    expect(evaluateFilterExpression(b.gt('year', 2020).build(), { year: 2021 })).toBe(true);
    expect(evaluateFilterExpression(b.gt('year', 2020).build(), { year: 2019 })).toBe(false);
    expect(evaluateFilterExpression(b.lt('year', 2020).build(), { year: 2019 })).toBe(true);
    expect(evaluateFilterExpression(b.lt('year', 2020).build(), { year: 2021 })).toBe(false);
    const orExp = b.or(b.eq('country', 'UK'), b.eq('country', 'US')).build();
    expect(evaluateFilterExpression(orExp, { country: 'US' })).toBe(true);
    expect(evaluateFilterExpression(orExp, { country: 'FR' })).toBe(false);
    const grouped = b.group(b.eq('active', true)).build();
    expect(evaluateFilterExpression(grouped, { active: true })).toBe(true);
  });

  test('ISNULL / NOT', () => {
    expect(evaluateFilterExpression(b.isNull('x').build(), {})).toBe(true);
    expect(evaluateFilterExpression(b.isNotNull('x').build(), { x: 1 })).toBe(true);
    expect(
      evaluateFilterExpression(b.not(b.eq('active', true)).build(), {
        active: false,
      }),
    ).toBe(true);
  });

  test('LTE and grouped operand nested under AND', () => {
    expect(evaluateFilterExpression(b.lte('year', 2020).build(), { year: 2020 })).toBe(true);
    expect(evaluateFilterExpression(b.lte('year', 2020).build(), { year: 2021 })).toBe(false);
    const nested = b.and(b.group(b.eq('active', true)), b.eq('country', 'UK')).build();
    expect(evaluateFilterExpression(nested, { active: true, country: 'UK' })).toBe(true);
    expect(evaluateFilterExpression(nested, { active: false, country: 'UK' })).toBe(false);
  });

  test('numeric-looking strings compare numerically', () => {
    const exp = b.gt('year', '5').build();
    expect(evaluateFilterExpression(exp, { year: '10' })).toBe(true);
    expect(evaluateFilterExpression(exp, { year: '2' })).toBe(false);
  });

  test('IN coerces a non-array filter value into a single-element list', () => {
    const exp = filterExpression('IN', filterKey('genre'), filterValue('drama'));
    expect(evaluateFilterExpression(exp, { genre: 'drama' })).toBe(true);
    expect(evaluateFilterExpression(exp, { genre: 'comedy' })).toBe(false);
  });

  test('quoted metadata keys are unwrapped before lookup', () => {
    const exp = filterExpression('EQ', filterKey(`'country'`), filterValue('UK'));
    expect(evaluateFilterExpression(exp, { country: 'UK' })).toBe(true);
  });

  test('max evaluation depth is enforced', () => {
    const andExp = b.and(b.eq('a', 1), b.eq('b', 2)).build();
    expect(() => evaluateFilterExpression(andExp, { a: 1, b: 2 }, 0)).toThrow(/max depth/);
    expect(() => evaluateFilterExpression(andExp, { a: 1, b: 2 }, -1)).toThrow(/max depth/);
  });

  test('rejects unknown expression types', () => {
    expect(() => evaluateFilterExpression({ type: 'NOOP' } as never, {})).toThrow(
      /Unsupported expression type/,
    );
  });

  test('malformed AST nodes raise descriptive errors', () => {
    expect(() =>
      evaluateFilterExpression(
        filterExpression('EQ', null as unknown as ReturnType<typeof filterKey>, filterValue(1)),
        {},
      ),
    ).toThrow(/requires a left operand/);

    expect(() => evaluateFilterExpression(filterExpression('EQ', filterKey('x')), {})).toThrow(
      /requires a right operand/,
    );

    expect(() =>
      evaluateFilterExpression(
        filterExpression(
          'EQ',
          filterValue(1) as unknown as ReturnType<typeof filterKey>,
          filterValue(1),
        ),
        {},
      ),
    ).toThrow(/Expected filter key operand/);

    expect(() =>
      evaluateFilterExpression(
        filterExpression(
          'EQ',
          filterKey('x'),
          filterKey('y') as unknown as ReturnType<typeof filterValue>,
        ),
        { x: 1 },
      ),
    ).toThrow(/Expected filter value operand/);

    expect(() =>
      evaluateFilterExpression(
        filterExpression(
          'AND',
          { kind: 'bogus' } as unknown as ReturnType<typeof filterKey>,
          filterValue(1),
        ),
        {},
      ),
    ).toThrow(/Unsupported boolean operand/);

    expect(() =>
      evaluateFilterExpression(
        filterExpression(
          'BOGUS' as unknown as Parameters<typeof filterExpression>[0],
          filterKey('x'),
          filterValue(1),
        ),
        { x: 1 },
      ),
    ).toThrow(/Unsupported expression type/);
  });

  test('text parser', () => {
    const exp = parseFilterExpression("country == 'UK' && year >= 2020 && isActive == true");
    expect(
      evaluateFilterExpression(exp, {
        country: 'UK',
        year: 2020,
        isActive: true,
      }),
    ).toBe(true);
    expect(
      evaluateFilterExpression(exp, {
        country: 'UK',
        year: 2019,
        isActive: true,
      }),
    ).toBe(false);
  });

  test('text parser rejects excessive nesting depth', () => {
    const deep = `${'('.repeat(70)}country == 'UK'${')'.repeat(70)}`;
    expect(() => parseFilterExpression(deep, 16)).toThrow(/max depth/);
  });

  test('text parser OR and grouping', () => {
    const exp = parseFilterExpression("(country == 'BG' || country == 'UK') AND year > 2000");
    expect(evaluateFilterExpression(exp, { country: 'BG', year: 2001 })).toBe(true);
    expect(evaluateFilterExpression(exp, { country: 'US', year: 2001 })).toBe(false);
  });

  test('text parser IN / NOT IN with populated and empty lists', () => {
    const inExp = parseFilterExpression("genre IN ['comedy', 'drama']");
    expect(evaluateFilterExpression(inExp, { genre: 'drama' })).toBe(true);
    expect(evaluateFilterExpression(inExp, { genre: 'horror' })).toBe(false);

    const ninExp = parseFilterExpression("city NIN ['Sofia', 'Plovdiv']");
    expect(evaluateFilterExpression(ninExp, { city: 'Varna' })).toBe(true);

    const notInExp = parseFilterExpression("city NOT IN ['Sofia']");
    expect(evaluateFilterExpression(notInExp, { city: 'Varna' })).toBe(true);
    expect(evaluateFilterExpression(notInExp, { city: 'Sofia' })).toBe(false);

    const emptyListExp = parseFilterExpression('tag IN []');
    expect(evaluateFilterExpression(emptyListExp, { tag: 'anything' })).toBe(false);
  });

  test('text parser IS NOT NULL, NE/LT/LTE operators, and null/false/number literals', () => {
    expect(evaluateFilterExpression(parseFilterExpression('x IS NOT NULL'), { x: 1 })).toBe(true);
    expect(evaluateFilterExpression(parseFilterExpression('x IS NOT NULL'), {})).toBe(false);
    expect(evaluateFilterExpression(parseFilterExpression('year != 2020'), { year: 2021 })).toBe(
      true,
    );
    expect(evaluateFilterExpression(parseFilterExpression('year < 2020'), { year: 2019 })).toBe(
      true,
    );
    expect(evaluateFilterExpression(parseFilterExpression('year <= 2020'), { year: 2020 })).toBe(
      true,
    );
    expect(evaluateFilterExpression(parseFilterExpression('flag == null'), { flag: null })).toBe(
      true,
    );
    expect(evaluateFilterExpression(parseFilterExpression('flag == false'), { flag: false })).toBe(
      true,
    );
    expect(evaluateFilterExpression(parseFilterExpression('score == -1.5'), { score: -1.5 })).toBe(
      true,
    );
  });

  test('isFilterExpression distinguishes expressions from other operands', () => {
    expect(isFilterExpression(filterExpression('EQ', filterKey('x'), filterValue(1)))).toBe(true);
    expect(isFilterExpression(filterKey('x'))).toBe(false);
    expect(isFilterExpression(filterValue(1))).toBe(false);
    expect(isFilterExpression(null)).toBe(false);
    expect(isFilterExpression(undefined)).toBe(false);
  });

  test('text parser surfaces syntax errors for malformed input', () => {
    expect(() => parseFilterExpression('country ===')).toThrow();
    expect(() => parseFilterExpression("country == 'UK' extra")).toThrow(/trailing/);
    expect(() => parseFilterExpression('country IS')).toThrow(/Expected keyword/);
    expect(() => parseFilterExpression('(country == 1')).toThrow(/Expected/);
    expect(() => parseFilterExpression('123')).toThrow(/Expected identifier/);
  });
});

describe('SimpleVectorStore', () => {
  test('add and similarity search ranks relevant docs', async () => {
    const store = SimpleVectorStore.of(new FakeEmbeddingModel());
    await store.add([
      textDocument(
        'Yorktown is in Virginia near the Chesapeake.',
        {
          source: 'a',
        },
        'd1',
      ),
      textDocument('The Eiffel Tower is in Paris, France.', { source: 'b' }, 'd2'),
      textDocument('Mount Fuji is a volcano in Japan.', { source: 'c' }, 'd3'),
    ]);
    expect(store.size).toBe(3);

    const hits = await store.similaritySearch(
      searchRequest({ query: 'Where is Yorktown?', topK: 2 }),
    );
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0]?.id).toBe('d1');
    expect(hits[0]?.score).toBeGreaterThan(0);
  });

  test('builder() supports a custom name via the fluent SimpleVectorStoreBuilder', async () => {
    const store = SimpleVectorStore.builder(new FakeEmbeddingModel()).name('custom-store').build();
    expect(store.name).toBe('custom-store');
    await store.add([textDocument('hello', {}, 'a')]);
    expect(store.size).toBe(1);
  });

  test('metadata filter restricts results', async () => {
    const store = SimpleVectorStore.of(new FakeEmbeddingModel());
    await store.add([
      textDocument('Cat fact one', { animal: 'cat' }, 'c1'),
      textDocument('Dog fact one', { animal: 'dog' }, 'd1'),
      textDocument('Cat fact two', { animal: 'cat' }, 'c2'),
    ]);

    const hits = await store.similaritySearch(
      searchRequest({
        query: 'animal facts',
        topK: 10,
        filterExpression: "animal == 'cat'",
      }),
    );
    expect(hits.every((h) => h.metadata.animal === 'cat')).toBe(true);
    expect(hits).toHaveLength(2);
  });

  test('delete by id and filter', async () => {
    const store = SimpleVectorStore.of(new FakeEmbeddingModel());
    await store.add([
      textDocument('one', { keep: false }, 'a'),
      textDocument('two', { keep: true }, 'b'),
    ]);
    await store.delete(['a']);
    expect(store.size).toBe(1);

    await store.add([textDocument('three', { keep: false }, 'c')]);
    await store.deleteByFilter?.(new FilterExpressionBuilder().eq('keep', false).build());
    expect(store.size).toBe(1);
    const remaining = await store.similaritySearchQuery?.('two');
    expect(remaining.some((d) => d.id === 'b')).toBe(true);
  });

  test('similarity threshold filters low scores', async () => {
    const store = SimpleVectorStore.of(new FakeEmbeddingModel());
    await store.add([textDocument('zzzz unrelated tokens', {}, 'z')]);
    const hits = await store.similaritySearch(
      searchRequest({
        query: 'Yorktown Virginia history',
        topK: 5,
        similarityThreshold: 0.99,
      }),
    );
    // Unrelated doc should not pass a very high threshold
    expect(hits.length).toBe(0);
  });

  test('stores Document.embedding and searches SearchRequest.queryEmbedding', async () => {
    const store = SimpleVectorStore.of(new PrecomputedEmbeddingModel(2));
    await store.add([
      document({ id: 'east', text: 'east', embedding: [1, 0] }),
      document({ id: 'north', text: 'north', embedding: [0, 1] }),
    ]);
    expect(await store.get('east')).toMatchObject({ id: 'east', embedding: [1, 0] });
    expect(await store.get('missing')).toBeNull();
    const hits = await store.similaritySearch(searchRequest({ queryEmbedding: [0, 1], topK: 1 }));
    expect(hits[0]?.id).toBe('north');
    expect(() => new PrecomputedEmbeddingModel().embed('noop')).toThrow(/queryEmbedding/);
    expect(() => new PrecomputedEmbeddingModel().embedDocument(textDocument('x'))).toThrow(
      /Document.embedding/,
    );
    expect(() =>
      new PrecomputedEmbeddingModel(2).embedDocument(document({ text: 'x', embedding: [1] })),
    ).toThrow(/dimension mismatch/);
  });
});

describe('ContextualQueryAugmenter', () => {
  test('injects context into prompt', () => {
    const augmenter = ContextualQueryAugmenter.builder();
    const augmented = augmenter.augment(query('Where is Yorktown?'), [
      textDocument('Yorktown is in Virginia.', {}, 'd1'),
    ]);
    expect(augmented.text).toContain('Yorktown is in Virginia.');
    expect(augmented.text).toContain('Where is Yorktown?');
  });

  test('empty context without allowEmptyContext uses empty template', () => {
    const augmenter = ContextualQueryAugmenter.builder({
      allowEmptyContext: false,
    });
    const augmented = augmenter.augment(query('q'), []);
    expect(augmented.text.toLowerCase()).toContain('knowledge');
  });

  test('allowEmptyContext returns original', () => {
    const augmenter = ContextualQueryAugmenter.builder({
      allowEmptyContext: true,
    });
    const original = query('q');
    expect(augmenter.augment(original, []).text).toBe('q');
  });
});

describe('RetrievalAugmentationAdvisor', () => {
  test('end-to-end RAG augments user message before model call', async () => {
    const store = SimpleVectorStore.of(new FakeEmbeddingModel());
    await store.add([
      textDocument('Yorktown is a historic town in Virginia.', { source: 'wiki' }, 'd1'),
      textDocument('Bananas are yellow fruit.', { source: 'wiki' }, 'd2'),
    ]);

    const model = new RecordingChatModel(new FakeChatModel('Yorktown is in Virginia.'));
    const rag = RetrievalAugmentationAdvisor.builder({
      documentRetriever: VectorStoreDocumentRetriever.builder({
        vectorStore: store,
        topK: 2,
      }),
    });

    const content = await ChatClient.builder(model)
      .defaultAdvisors(rag)
      .build()
      .prompt()
      .user('Where is Yorktown?')
      .call()
      .content();

    expect(content).toBe('Yorktown is in Virginia.');
    expect(model.calls).toHaveLength(1);
    const promptText = model.calls[0]?.getUserMessage().text ?? '';
    expect(promptText).toContain('Yorktown is a historic town in Virginia.');
    expect(promptText).toContain('Where is Yorktown?');
  });

  test('stream() path augments the request and re-attaches document context on every chunk', async () => {
    const store = SimpleVectorStore.of(new FakeEmbeddingModel());
    await store.add([textDocument('Yorktown is a historic town in Virginia.', {}, 'd1')]);

    const model = new FakeChatModel('Yorktown is in Virginia.');
    const rag = RetrievalAugmentationAdvisor.builder({
      documentRetriever: VectorStoreDocumentRetriever.builder({
        vectorStore: store,
        topK: 1,
      }),
    });

    const client = ChatClient.builder(model).defaultAdvisors(rag).build();

    const chunks: string[] = [];
    let lastResponse: ChatClientResponse | undefined;
    for await (const response of client
      .prompt()
      .user('Where is Yorktown?')
      .stream()
      .chatClientResponse()) {
      lastResponse = response;
      if (response.chatResponse?.content) chunks.push(response.chatResponse.content);
    }

    expect(chunks.at(-1)).toBe('Yorktown is in Virginia.');
    expect(lastResponse?.context.get(RAG_DOCUMENT_CONTEXT)).toBeDefined();
    expect(model.calls[0]?.getUserMessage().text ?? '').toContain(
      'Yorktown is a historic town in Virginia.',
    );
  });

  test('stores documents in context under RAG_DOCUMENT_CONTEXT', async () => {
    const store = SimpleVectorStore.of(new FakeEmbeddingModel());
    await store.add([textDocument('Ada Lovelace wrote early programs.', {}, 'd1')]);

    const client = ChatClient.builder(new FakeChatModel('Ada.'))
      .defaultAdvisors(
        RetrievalAugmentationAdvisor.builder({
          documentRetriever: VectorStoreDocumentRetriever.builder({
            vectorStore: store,
            topK: 1,
          }),
        }),
      )
      .build();

    const response = await client.prompt().user('Who is Ada?').call().chatClientResponse();

    const docs = response.context.get(RAG_DOCUMENT_CONTEXT) as { id: string }[] | undefined;
    expect(docs).toBeDefined();
    expect(docs?.length).toBeGreaterThan(0);
    expect(docs?.[0]?.id).toBe('d1');
    expect(response.chatResponse?.metadata[RAG_DOCUMENT_CONTEXT]).toBeDefined();
  });

  test('query transformer can rewrite the search text', async () => {
    const store = SimpleVectorStore.of(new FakeEmbeddingModel());
    await store.add([textDocument('The capital of France is Paris.', {}, 'd1')]);

    const model = new RecordingChatModel(new FakeChatModel('Paris.'));
    const rag = RetrievalAugmentationAdvisor.builder({
      documentRetriever: VectorStoreDocumentRetriever.builder({
        vectorStore: store,
        topK: 1,
      }),
      queryTransformers: [
        {
          transform: (q) =>
            query({
              text: 'capital of France Paris',
              history: q.history,
              context: q.context,
            }),
        },
      ],
    });

    await ChatClient.builder(model)
      .defaultAdvisors(rag)
      .build()
      .prompt()
      .user('???')
      .call()
      .content();

    const user = model.calls[0]?.getUserMessage().text ?? '';
    // Augmentation still uses original query text in template, with retrieved docs
    expect(user).toContain('The capital of France is Paris.');
  });
});

describe('VectorStoreDocumentRetriever filter expressions', () => {
  test('a fixed (non-function) filterExpression option is applied on every retrieve', async () => {
    const store = SimpleVectorStore.of(new FakeEmbeddingModel());
    await store.add([
      textDocument('Yorktown wiki doc', { source: 'wiki' }, 'd1'),
      textDocument('Yorktown blog doc', { source: 'blog' }, 'd2'),
    ]);

    const retriever = VectorStoreDocumentRetriever.builder({
      vectorStore: store,
      topK: 5,
      filterExpression: filterExpression('EQ', filterKey('source'), filterValue('wiki')),
    });

    const results = await retriever.retrieve(query('Yorktown'));
    expect(results.map((d) => d.id)).toEqual(['d1']);
  });

  test('a per-query context filter expression overrides the default filter', async () => {
    const store = SimpleVectorStore.of(new FakeEmbeddingModel());
    await store.add([
      textDocument('Yorktown wiki doc', { source: 'wiki' }, 'd1'),
      textDocument('Yorktown blog doc', { source: 'blog' }, 'd2'),
    ]);

    const retriever = VectorStoreDocumentRetriever.builder({
      vectorStore: store,
      topK: 5,
    });

    const withObjectFilter = await retriever.retrieve(
      query({
        text: 'Yorktown',
        context: {
          vector_store_filter_expression: filterExpression(
            'EQ',
            filterKey('source'),
            filterValue('blog'),
          ),
        },
      }),
    );
    expect(withObjectFilter.map((d) => d.id)).toEqual(['d2']);

    const withTextFilter = await retriever.retrieve(
      query({
        text: 'Yorktown',
        context: { vector_store_filter_expression: "source == 'wiki'" },
      }),
    );
    expect(withTextFilter.map((d) => d.id)).toEqual(['d1']);
  });
});
