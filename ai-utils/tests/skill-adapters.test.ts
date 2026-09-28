import { describe, expect, test } from 'bun:test';
import { chatClientRequest, Prompt, ScriptedChatModel } from '@di-framework/ai';
import type { SkillEmbedder } from '../src/index.ts';
import {
  agentSkill,
  aggregateSkillChunkMatches,
  asyncSkillsTool,
  benchmarkSkillVectorSearch,
  buildSkillsIndex,
  createSkillsAgentAsync,
  createSkillsAgentBundleAsync,
  createSkillsToolbox,
  createSkillsToolboxAsync,
  InMemorySkillCatalogStore,
  InMemorySkillVectorSearch,
  LocalSkillCatalogStore,
  LocalSkillIndexWriter,
  LocalSkillVectorSearch,
  runSkillAdapterOperation,
  SkillAdapterError,
  SkillsAgent,
  SkillsIndex,
  SkillsRetrievalAdvisor,
  SkillsToolbox,
  skillsTool,
} from '../src/index.ts';

const embedder: SkillEmbedder = {
  id: 'local@test',
  model: 'local',
  revision: 'test',
  async embed(texts) {
    return texts.map((text) =>
      text.includes('alpha') ? new Float32Array([1, 0]) : new Float32Array([0, 1]),
    );
  },
  async split(text) {
    return [text];
  },
};

describe('platform-neutral skill adapters', () => {
  test('catalog isolates namespaces and validates activation versions', async () => {
    const alpha = agentSkill({ name: 'review', description: 'Alpha review', content: 'alpha' });
    const beta = agentSkill({ name: 'review', description: 'Beta review', content: 'beta' });
    const store = new InMemorySkillCatalogStore(
      [
        {
          descriptor: { name: 'review', sourceHash: 'a', version: '1', namespace: 'a' },
          skill: alpha,
        },
        {
          descriptor: { name: 'review', sourceHash: 'b', version: '2', namespace: 'b' },
          skill: beta,
        },
      ],
      { a: 'catalog-a', b: 'catalog-b' },
    );

    expect(await store.list({ namespace: 'a' })).toEqual([
      { name: 'review', sourceHash: 'a', version: '1', namespace: 'a' },
    ]);
    expect((await store.load('review', { namespace: 'b', expectedVersion: '2' }))?.content).toBe(
      'beta',
    );
    await expect(
      store.load('review', { namespace: 'b', expectedVersion: 'old' }),
    ).rejects.toMatchObject({ code: 'STALE_CATALOG' });
    expect(await store.version({ namespace: 'a' })).toBe('catalog-a');
    expect(await store.health({ namespace: 'b' })).toMatchObject({
      status: 'ready',
      checkedVersion: 'catalog-b',
    });
    expect(
      await new InMemorySkillCatalogStore([
        { descriptor: { name: 'review', sourceHash: 'generated' }, skill: alpha },
      ]).version(),
    ).toMatch(/^fnv1a32:/);
  });

  test('exact search enforces ready catalog and model metadata', async () => {
    const search = new InMemorySkillVectorSearch();
    const receipt = await search.replace({
      metadata: {
        indexVersion: 'index-1',
        catalogVersion: 'catalog-1',
        dimensions: 2,
        model: 'model',
        revision: 'revision',
        embedderId: 'model@revision',
        scoring: 'cosine',
      },
      vectors: [
        {
          name: 'beta',
          description: 'Beta',
          chunk: 0,
          source: 'document',
          embedding: [1, 0],
        },
        {
          name: 'alpha',
          description: 'Alpha',
          chunk: 0,
          source: 'document',
          embedding: [1, 0],
        },
      ],
    });

    expect(receipt).toMatchObject({ ready: true, writtenVectors: 2 });
    expect(await search.metadata()).toMatchObject({ indexVersion: 'index-1' });
    expect(await search.health()).toMatchObject({ status: 'ready' });
    expect(
      await search.query([1, 0], {
        limit: 1,
        catalogVersion: 'catalog-1',
        model: 'model',
      }),
    ).toEqual([
      {
        name: 'alpha',
        description: 'Alpha',
        score: 1,
        chunk: 0,
        source: 'document',
      },
    ]);
    await expect(search.query([1, 0], { catalogVersion: 'stale' })).rejects.toMatchObject({
      code: 'STALE_CATALOG',
    });
    await expect(search.query([1, 0], { model: 'other' })).rejects.toMatchObject({
      code: 'MODEL_MISMATCH',
    });
    expect(new SkillAdapterError('TIMEOUT', 'late')).toMatchObject({
      name: 'SkillAdapterError',
      code: 'TIMEOUT',
    });
    await search.upsert?.({
      metadata: {
        indexVersion: 'index-2',
        catalogVersion: 'catalog-1',
        dimensions: 2,
        model: 'model',
        revision: 'revision',
        embedderId: 'model@revision',
        scoring: 'cosine',
      },
      vectors: [
        {
          name: 'gamma',
          description: 'Gamma',
          chunk: 0,
          source: 'document',
          embedding: [0, 1],
        },
      ],
    });
    expect((await search.query([0, 1], { limit: 1 }))[0]?.name).toBe('gamma');
    expect(await new InMemorySkillVectorSearch().health()).toMatchObject({ status: 'not-ready' });
    const initialized = new InMemorySkillVectorSearch([
      {
        metadata: {
          indexVersion: 'array-index',
          catalogVersion: 'array-catalog',
          dimensions: 1,
          scoring: 'cosine',
        },
        vectors: [],
      },
    ]);
    expect(await initialized.metadata()).toMatchObject({ indexVersion: 'array-index' });
  });
});

describe('asynchronous skill activation', () => {
  test('catalog bodies stay lazy until the selected tool executes', async () => {
    const skill = agentSkill({
      name: 'remote-review',
      description: 'Review remote code',
      content: 'SECRET REMOTE BODY',
    });
    let loads = 0;
    const base = new InMemorySkillCatalogStore(
      [
        {
          descriptor: {
            name: skill.name,
            description: skill.description,
            sourceHash: 'hash',
            version: 'v1',
          },
          skill,
        },
      ],
      { '': 'catalog-v1' },
    );
    const store = {
      capabilities: base.capabilities,
      list: base.list.bind(base),
      version: base.version.bind(base),
      health: base.health.bind(base),
      async load(name: string, options?: Parameters<typeof base.load>[1]) {
        loads++;
        return base.load(name, options);
      },
    };
    const descriptors = await store.list();
    const tool = asyncSkillsTool({ descriptors, catalogStore: store });

    expect(tool.toolDefinition.description).toContain('Review remote code');
    expect(tool.toolDefinition.description).not.toContain('SECRET REMOTE BODY');
    expect(loads).toBe(0);
    expect(await tool.call(JSON.stringify({ command: 'remote-review' }))).toContain(
      'SECRET REMOTE BODY',
    );
    expect(loads).toBe(1);
  });

  test('async toolbox uses independent catalog and vector adapters', async () => {
    const alpha = agentSkill({
      name: 'alpha',
      description: 'Handle alpha tasks',
      content: 'ALPHA PRIVATE BODY',
    });
    const beta = agentSkill({
      name: 'beta',
      description: 'Handle beta tasks',
      content: 'BETA PRIVATE BODY',
    });
    const catalogStore = new InMemorySkillCatalogStore(
      [alpha, beta].map((skill) => ({
        descriptor: {
          name: skill.name,
          description: skill.description,
          sourceHash: `${skill.name}-hash`,
          version: `${skill.name}-v1`,
        },
        skill,
      })),
      { '': 'catalog-v1' },
    );
    const vectorSearch = new InMemorySkillVectorSearch({
      metadata: {
        indexVersion: 'index-v1',
        catalogVersion: 'catalog-v1',
        dimensions: 2,
        model: embedder.model,
        revision: embedder.revision,
        embedderId: embedder.id,
        scoring: 'cosine',
      },
      vectors: [
        {
          name: 'alpha',
          description: 'Handle alpha tasks',
          chunk: 0,
          source: 'document',
          embedding: [1, 0],
        },
        {
          name: 'beta',
          description: 'Handle beta tasks',
          chunk: 0,
          source: 'document',
          embedding: [0, 1],
        },
      ],
    });
    const options = {
      workspace: process.cwd(),
      semanticDiscovery: { catalogStore, vectorSearch, embedder, limit: 1 },
      todos: false,
      list: false,
      glob: false,
      grep: false,
    } as const;
    const toolbox = await createSkillsToolboxAsync(options);
    expect(toolbox.skills).toEqual([]);
    expect(toolbox.descriptors).toHaveLength(2);
    const request = chatClientRequest(
      new Prompt('please do an alpha task', { toolCallbacks: toolbox.tools }),
    );
    const selected = await toolbox.retrievalAdvisor?.before(request);
    const skillTool = selected?.prompt.options?.toolCallbacks?.find(
      (tool) => tool.toolDefinition.name === 'Skill',
    );
    expect(skillTool?.toolDefinition.description).toContain('<name>alpha</name>');
    expect(skillTool?.toolDefinition.description).not.toContain('<name>beta</name>');
    expect(skillTool?.toolDefinition.description).not.toContain('PRIVATE BODY');
    const availableCatalog = skillTool?.toolDefinition.description.match(
      /<available_skills>\n<skill>[\s\S]*<\/available_skills>/,
    )?.[0];
    expect(availableCatalog).toMatchInlineSnapshot(`
      "<available_skills>
      <skill>
        <name>alpha</name>
        <description>Handle alpha tasks</description>
      </skill>
      </available_skills>"
    `);
    expect(JSON.stringify(selected?.prompt)).not.toContain('index-v1');
    expect(JSON.stringify(selected?.prompt)).not.toContain('alpha-hash');
    expect(await skillTool?.call(JSON.stringify({ command: 'alpha' }))).toContain(
      'ALPHA PRIVATE BODY',
    );
    expect((await SkillsToolbox.ofAsync(options)).descriptors).toHaveLength(2);
    const builder = SkillsToolbox.builder()
      .workspace(process.cwd())
      .sourceMode('replace')
      .semanticDiscovery(options.semanticDiscovery)
      .todos(false)
      .list(false)
      .glob(false)
      .grep(false);
    expect((await builder.buildAsync()).descriptors).toHaveLength(2);
    expect(await builder.buildToolsAsync()).not.toHaveLength(0);
    expect(() => createSkillsToolbox(options)).toThrow(/Asynchronous catalog stores/);
    expect(
      (
        await createSkillsToolboxAsync({
          skills: [alpha],
          sourceMode: 'replace',
          semanticDiscovery: false,
          todos: false,
        })
      ).skills,
    ).toEqual([alpha]);

    const degradedCatalog = {
      capabilities: catalogStore.capabilities,
      list: catalogStore.list.bind(catalogStore),
      load: catalogStore.load.bind(catalogStore),
      version: catalogStore.version.bind(catalogStore),
      async health() {
        return { status: 'degraded' as const, message: 'catalog unavailable' };
      },
    };
    await expect(
      createSkillsToolboxAsync({
        ...options,
        semanticDiscovery: { ...options.semanticDiscovery, catalogStore: degradedCatalog },
      }),
    ).rejects.toMatchObject({ code: 'NOT_READY' });

    await expect(
      createSkillsToolboxAsync({
        ...options,
        semanticDiscovery: {
          ...options.semanticDiscovery,
          vectorSearch: new InMemorySkillVectorSearch(),
        },
      }),
    ).rejects.toMatchObject({ code: 'NOT_READY' });

    const chatModel = new ScriptedChatModel([]);
    await expect(createSkillsAgentAsync({ ...options, chatModel })).resolves.toBeDefined();
    const bundle = await createSkillsAgentBundleAsync({ ...options, chatModel });
    expect(bundle.toolbox.descriptors).toHaveLength(2);
    await expect(SkillsAgent.ofAsync({ ...options, chatModel })).resolves.toBeDefined();
    const agentBuilder = SkillsAgent.builder()
      .chatModel(chatModel)
      .workspace(process.cwd())
      .sourceMode('replace')
      .semanticDiscovery(options.semanticDiscovery)
      .todos(false)
      .list(false)
      .glob(false)
      .grep(false);
    await expect(agentBuilder.buildAsync()).resolves.toBeDefined();
    expect((await agentBuilder.buildBundleAsync()).toolbox.descriptors).toHaveLength(2);
    expect(
      SkillsAgent.builder()
        .chatModel(chatModel)
        .sourceMode('replace')
        .addSkill(alpha)
        .todos(false)
        .build(),
    ).toBeDefined();
  });

  test('missing bodies, timeouts, and unavailable adapters fail closed', async () => {
    const descriptor = { name: 'missing', description: 'Missing body', sourceHash: 'hash' };
    const tool = asyncSkillsTool({
      descriptors: [descriptor],
      catalogStore: new InMemorySkillCatalogStore([]),
    });
    await expect(tool.call(JSON.stringify({ command: 'missing' }))).rejects.toThrow(/no body/);
    await expect(
      runSkillAdapterOperation('Slow catalog', () => new Promise(() => {}), 1),
    ).rejects.toMatchObject({ code: 'TIMEOUT' });
    await expect(new InMemorySkillVectorSearch().query([1])).rejects.toMatchObject({
      code: 'NOT_READY',
    });
  });

  test('async agent factories validate adapter readiness', async () => {
    const skill = agentSkill({ name: 'remote', description: 'Remote tasks', content: 'body' });
    const catalogStore = new InMemorySkillCatalogStore(
      [
        {
          descriptor: {
            name: skill.name,
            description: skill.description,
            sourceHash: 'remote-hash',
          },
          skill,
        },
      ],
      { '': 'catalog' },
    );
    const vectorSearch = new InMemorySkillVectorSearch({
      metadata: {
        indexVersion: 'remote-index',
        catalogVersion: 'catalog',
        dimensions: 2,
        model: embedder.model,
        revision: embedder.revision,
        embedderId: embedder.id,
        scoring: 'cosine',
      },
      vectors: [
        {
          name: 'remote',
          description: 'Remote tasks',
          chunk: 0,
          source: 'document',
          embedding: [0, 1],
        },
      ],
    });
    const options = {
      workspace: process.cwd(),
      directories: [],
      semanticDiscovery: { catalogStore, vectorSearch, embedder, timeoutMs: 50 },
      todos: false,
      list: false,
      glob: false,
      grep: false,
    } as const;
    const chatModel = new ScriptedChatModel([]);
    await expect(createSkillsAgentAsync({ ...options, chatModel })).resolves.toBeDefined();
    await expect(createSkillsAgentBundleAsync({ ...options, chatModel })).resolves.toMatchObject({
      toolbox: { descriptors: [expect.anything()] },
    });
    await expect(SkillsAgent.ofAsync({ ...options, chatModel })).resolves.toBeDefined();
    const builder = SkillsAgent.builder()
      .chatModel(chatModel)
      .workspace(process.cwd())
      .sourceMode('replace')
      .semanticDiscovery(options.semanticDiscovery)
      .todos(false)
      .list(false)
      .glob(false)
      .grep(false);
    await expect(builder.buildAsync()).resolves.toBeDefined();
    await expect(builder.buildBundleAsync()).resolves.toMatchObject({
      toolbox: { descriptors: [expect.anything()] },
    });
    await expect(
      createSkillsToolboxAsync({
        ...options,
        semanticDiscovery: {
          ...options.semanticDiscovery,
          vectorSearch: new InMemorySkillVectorSearch(),
        },
      }),
    ).rejects.toMatchObject({ code: 'NOT_READY' });

    const invalidSkill = agentSkill({
      name: 'Invalid Name',
      description: 'Invalid descriptor',
      content: 'body',
    });
    const invalidCatalog = new InMemorySkillCatalogStore([
      {
        descriptor: {
          name: invalidSkill.name,
          description: invalidSkill.description,
          sourceHash: 'invalid',
        },
        skill: invalidSkill,
      },
    ]);
    await expect(
      createSkillsToolboxAsync({
        ...options,
        semanticDiscovery: { ...options.semanticDiscovery, catalogStore: invalidCatalog },
      }),
    ).rejects.toMatchObject({ code: 'INVALID_RESPONSE' });
  });
});

describe('local adapters', () => {
  test('filesystem-compatible catalog and JSONL search implement the contracts', async () => {
    const skills = [
      agentSkill({ name: 'alpha', description: 'alpha routing', content: 'alpha body' }),
      agentSkill({ name: 'beta', description: 'beta routing', content: 'beta body' }),
    ];
    const catalog = new LocalSkillCatalogStore(skills);
    const outputFile = `${process.env.TMPDIR ?? '/tmp'}/local-adapter-${crypto.randomUUID()}.jsonl`;
    await buildSkillsIndex({ skills, outputFile, threshold: 1, embedder });
    const search = new LocalSkillVectorSearch(outputFile);

    expect((await catalog.list()).map(({ name }) => name)).toEqual(['alpha', 'beta']);
    expect((await catalog.load('alpha'))?.content).toBe('alpha body');
    expect(await catalog.load('missing')).toBeUndefined();
    expect(await catalog.version()).toBeTruthy();
    expect(await catalog.health()).toMatchObject({ status: 'ready' });
    await expect(catalog.load('alpha', { expectedVersion: 'stale' })).rejects.toMatchObject({
      code: 'STALE_CATALOG',
    });
    await expect(catalog.list({ namespace: 'remote' })).rejects.toThrow(/does not support/);
    expect((await search.query([1, 0], { limit: 1 }))[0]).toMatchObject({
      name: 'alpha',
      chunk: 0,
      score: 1,
    });
    expect(await search.health()).toMatchObject({ status: 'ready' });
  });

  test('index builders can write to a replacement adapter and return readiness', async () => {
    const writer = new InMemorySkillVectorSearch();
    const skills = [
      agentSkill({ name: 'alpha', description: 'Alpha tasks', content: 'alpha' }),
      agentSkill({ name: 'beta', description: 'Beta tasks', content: 'beta' }),
    ];
    const result = await SkillsIndex.builder()
      .addSkills(skills)
      .threshold(1)
      .embedder(embedder)
      .writer(writer)
      .build();

    expect(result.receipt).toMatchObject({ ready: true, writtenVectors: 2 });
    expect((await writer.query([0, 1], { limit: 1 }))[0]?.name).toBe('beta');
  });

  test('local writer groups chunks and object-backed search reports metadata', async () => {
    const outputFile = `${process.env.TMPDIR ?? '/tmp'}/local-writer-${crypto.randomUUID()}.jsonl`;
    const writer = new LocalSkillIndexWriter(outputFile);
    await writer.replace({
      metadata: {
        indexVersion: 'local-write',
        catalogVersion: 'catalog-write',
        dimensions: 2,
        model: 'local',
        revision: 'test',
        embedderId: 'local@test',
        scoring: 'cosine',
      },
      vectors: [
        {
          name: 'alpha',
          description: 'Alpha',
          documentHash: 'hash',
          chunk: 0,
          source: 'document',
          embedding: [1, 0],
        },
        {
          name: 'alpha',
          description: 'Alpha',
          documentHash: 'hash',
          chunk: 1,
          source: 'document',
          embedding: [0, 1],
        },
      ],
    });
    const loaded = new LocalSkillVectorSearch(outputFile).index;
    const search = new LocalSkillVectorSearch(loaded);
    expect(loaded.entries[0]?.chunks).toHaveLength(2);
    expect(await search.metadata()).toMatchObject({ catalogVersion: 'catalog-write', ready: true });
  });

  test('adapter retrieval shares aggregation and rejects partial backend results', async () => {
    const skills = [
      agentSkill({ name: 'alpha', description: 'Alpha tasks', content: 'alpha' }),
      agentSkill({ name: 'beta', description: 'Beta tasks', content: 'beta' }),
    ];
    const catalog = new InMemorySkillCatalogStore(
      skills.map((skill) => ({
        descriptor: {
          name: skill.name,
          description: skill.description,
          sourceHash: skill.name,
        },
        skill,
      })),
      { '': 'catalog' },
    );
    const vector = new InMemorySkillVectorSearch({
      metadata: {
        indexVersion: 'adapter',
        catalogVersion: 'catalog',
        dimensions: 2,
        model: embedder.model,
        revision: embedder.revision,
        embedderId: embedder.id,
        scoring: 'cosine',
      },
      vectors: [
        {
          name: 'alpha',
          description: 'Alpha tasks',
          chunk: 0,
          source: 'document',
          embedding: [1, 0],
        },
        {
          name: 'alpha',
          description: 'Alpha tasks',
          chunk: 1,
          source: 'document',
          embedding: [0, 1],
        },
        {
          name: 'beta',
          description: 'Beta tasks',
          chunk: 0,
          source: 'document',
          embedding: [0, 1],
        },
      ],
    });
    const advisor = new SkillsRetrievalAdvisor({
      skills,
      catalogStore: catalog,
      vectorSearch: vector,
      embedder,
      limit: 1,
    });
    const selected = await advisor.before(
      chatClientRequest(new Prompt('alpha request', { toolCallbacks: [skillsTool({ skills })] })),
    );
    expect(selected.context.get('skills_retrieval')).toEqual([
      expect.objectContaining({ name: 'alpha' }),
    ]);
    expect(
      aggregateSkillChunkMatches(
        [
          {
            name: 'alpha',
            description: 'Alpha',
            score: 0.4,
            chunk: 1,
            source: 'document',
          },
        ],
        1,
      )[0],
    ).toMatchObject({ score: 0.4, matchedChunk: 1 });
    expect(() =>
      aggregateSkillChunkMatches(
        [
          {
            name: 'unknown',
            description: 'Unknown',
            score: Number.NaN,
            chunk: -1,
            source: 'document',
          },
        ],
        1,
        new Set(skills.map((skill) => skill.name)),
      ),
    ).toThrow(/invalid chunk/);
    expect(() =>
      aggregateSkillChunkMatches(
        [
          {
            name: 'unknown',
            description: 'Unknown',
            score: 1,
            chunk: 0,
            source: 'document',
          },
        ],
        1,
        new Set(skills.map((skill) => skill.name)),
      ),
    ).toThrow(/unknown skill/);
  });

  test('performance reporter accepts quality results from the evaluation corpus', async () => {
    const search = new InMemorySkillVectorSearch({
      metadata: {
        indexVersion: 'benchmark',
        catalogVersion: 'catalog',
        dimensions: 2,
        scoring: 'cosine',
      },
      vectors: [
        {
          name: 'alpha',
          description: 'Alpha',
          chunk: 0,
          source: 'document',
          embedding: [1, 0],
        },
      ],
    });
    let clock = 0;
    const report = await benchmarkSkillVectorSearch({
      createSearch: () => search,
      cases: [{ vector: [1, 0], options: { limit: 1 } }],
      warmupTrials: 0,
      measuredTrials: 2,
      now: () => clock++,
      quality: {
        positiveTrials: 30,
        noSkillTrials: 2,
        recallAt1: 1,
        recallAt10: 1,
        meanReciprocalRank: 1,
        abstentionRate: 1,
        noSkillFalsePositiveRate: 0,
      },
    });
    expect(report).toMatchObject({
      schemaVersion: 1,
      measuredTrials: 2,
      coldInitializationMs: 1,
      vectorSearchMs: { p50: 1, p95: 1, mean: 1 },
      quality: { recallAt1: 1, recallAt10: 1, meanReciprocalRank: 1 },
    });
    const evaluationQuality = await benchmarkSkillVectorSearch({
      createSearch: () => search,
      cases: [{ vector: [1, 0] }],
      warmupTrials: 0,
      measuredTrials: 1,
      now: () => clock++,
      quality: {
        schemaVersion: 1,
        suite: 'awesome-copilot semantic retrieval baseline',
        corpus: { id: 'github/awesome-copilot', revision: 'pinned', skillCount: 408 },
        trialsPerCase: 1,
        caseCount: 30,
        metrics: {
          positiveTrials: 30,
          noSkillTrials: 0,
          recallAt1: 29 / 30,
          recallAt10: 1,
          meanReciprocalRank: 0.9708,
          abstentionRate: 0,
          noSkillFalsePositiveRate: 0,
        },
      },
    });
    expect(evaluationQuality.quality).toMatchObject({
      suite: 'awesome-copilot semantic retrieval baseline',
      corpus: { revision: 'pinned', skillCount: 408 },
      metrics: { positiveTrials: 30, recallAt10: 1, noSkillFalsePositiveRate: 0 },
    });
    expect(
      (
        await benchmarkSkillVectorSearch({
          createSearch: () => search,
          cases: [{ vector: [1, 0] }],
          measuredTrials: 1,
          now: () => clock++,
        })
      ).measuredTrials,
    ).toBe(1);
    await expect(
      benchmarkSkillVectorSearch({ createSearch: () => search, cases: [], warmupTrials: 0 }),
    ).rejects.toThrow(/benchmark case/);
    await expect(
      benchmarkSkillVectorSearch({
        createSearch: () => search,
        cases: [{ vector: [1, 0] }],
        warmupTrials: -1,
      }),
    ).rejects.toThrow(/non-negative/);
  });
});
