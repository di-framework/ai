import { describe, expect, test } from 'bun:test';
import { mkdtempSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  type CallAdvisorChain,
  type ChatModel,
  ChatResponse,
  chatClientRequest,
  chatClientResponse,
  Prompt,
  ScriptedChatModel,
  type StreamAdvisorChain,
  toolCall,
  toolCallResponse,
} from '@di-framework/ai';
import {
  agentSkill,
  assertSkillsIndexCurrent,
  buildSkillsIndex,
  cosineSimilarity,
  loadSkillsIndex,
  rankSkillsIndex,
  type SkillEmbedder,
  SkillsAgent,
  SkillsIndex,
  SkillsRetrievalAdvisor,
  type SkillTokenChunkOptions,
  scoreSkillsIndexEntry,
  searchSkillsIndex,
  skillsTool,
} from '../src/index.ts';

class TestEmbedder implements SkillEmbedder {
  readonly id = 'test-embedding-model@test-revision';
  readonly model = 'test-embedding-model';
  readonly revision = 'test-revision';
  calls = 0;
  readonly splitTexts: string[] = [];

  async embed(texts: readonly string[]): Promise<readonly Float32Array[]> {
    this.calls++;
    return texts.map((text) => {
      const lower = text.toLowerCase();
      if (lower.includes('pdf')) return new Float32Array([1, 0, 0]);
      if (lower.includes('postgres')) return new Float32Array([0, 1, 0]);
      return new Float32Array([0, 0, 1]);
    });
  }

  async split(text: string, options: SkillTokenChunkOptions): Promise<readonly string[]> {
    this.splitTexts.push(text);
    return [`${options.prefix ?? ''}${text}`];
  }
}

function testSkills() {
  return [
    agentSkill({
      name: 'pdf-reader',
      description: 'Read and extract tables from PDF documents.',
      content: 'Use the PDF parser.',
    }),
    agentSkill({
      name: 'postgres-reviewer',
      description: 'Review PostgreSQL schemas and queries.',
      content: 'Inspect indexes and row-level security.',
    }),
    agentSkill({
      name: 'weather-reporter',
      description: 'Report weather forecasts for a city.',
      content: 'Use the weather service.',
    }),
  ];
}

async function indexedFixture() {
  const directory = mkdtempSync(join(tmpdir(), 'skills-index-'));
  const outputFile = join(directory, 'skills.jsonl');
  const embedder = new TestEmbedder();
  const skills = testSkills();
  const result = await buildSkillsIndex({
    skills,
    outputFile,
    threshold: 2,
    retrievalLimit: 2,
    batchSize: 2,
    embedder,
  });
  return { directory, outputFile, embedder, skills, result, index: loadSkillsIndex(outputFile) };
}

describe('buildSkillsIndex', () => {
  test('SkillsIndex.builder exposes the complete build-time configuration', async () => {
    const directory = mkdtempSync(join(tmpdir(), 'skills-index-builder-'));
    const outputFile = join(directory, 'skills.jsonl');
    const embedder = new TestEmbedder();
    const progress: Array<[number, number]> = [];
    const [firstSkill, ...remainingSkills] = testSkills();
    if (!firstSkill) throw new Error('missing test skill');

    const builder = SkillsIndex.builder()
      .addSkill(firstSkill)
      .addSkills(remainingSkills.slice(0, 1))
      .addSkills(remainingSkills.slice(1))
      .addSkillsDirectories([])
      .addSkillsFiles([])
      .outputFile(outputFile)
      .threshold(2)
      .retrievalLimit(2)
      .batchSize(2)
      .chunkTokens(128)
      .chunkOverlapTokens(16)
      .embedder(embedder)
      .force()
      .onProgress((completed, total) => progress.push([completed, total]));

    expect(builder.toOptions()).toMatchObject({
      outputFile,
      threshold: 2,
      retrievalLimit: 2,
      batchSize: 2,
      chunkTokens: 128,
      chunkOverlapTokens: 16,
      force: true,
    });
    expect(
      SkillsIndex.builder().addSkillsDirectory('directory').addSkillsFile('skill.md').toOptions(),
    ).toMatchObject({ directories: ['directory'], files: ['skill.md'] });
    expect(await builder.build()).toMatchObject({ indexed: true, skillCount: 3, chunkCount: 3 });
    expect(progress).toEqual([
      [2, 3],
      [3, 3],
    ]);
  });

  test('skips embeddings at or below the catalog threshold', async () => {
    const directory = mkdtempSync(join(tmpdir(), 'skills-index-small-'));
    const outputFile = join(directory, 'skills.jsonl');
    const embedder = new TestEmbedder();
    const result = await buildSkillsIndex({
      skills: testSkills().slice(0, 2),
      outputFile,
      threshold: 2,
      embedder,
    });

    expect(result.indexed).toBe(false);
    expect(embedder.calls).toBe(0);
    const index = loadSkillsIndex(outputFile);
    expect(index.metadata.indexed).toBe(false);
    expect(index.entries).toEqual([]);
    expect(readFileSync(outputFile, 'utf8').trim().split('\n')).toHaveLength(1);
  });

  test('writes deterministic metadata and normalized vector records in batches', async () => {
    const fixture = await indexedFixture();
    expect(fixture.result).toMatchObject({
      indexed: true,
      skillCount: 3,
      chunkCount: 3,
      dimensions: 3,
    });
    expect(fixture.embedder.calls).toBe(2);
    expect(fixture.embedder.splitTexts[0]).toContain('name: "pdf-reader"');
    expect(fixture.embedder.splitTexts[0]).toContain('Use the PDF parser.');
    expect(fixture.index.metadata).toMatchObject({
      model: fixture.embedder.model,
      revision: fixture.embedder.revision,
      retrievalLimit: 2,
    });
    expect(fixture.index.entries.map((entry) => entry.name)).toEqual([
      'pdf-reader',
      'postgres-reviewer',
      'weather-reporter',
    ]);
    expect(fixture.index.entries[0]?.chunks).toHaveLength(1);
    expect(fixture.index.entries[0]?.chunks[0]?.embedding).toMatchObject({
      values: new Int8Array([127, 0, 0]),
    });
    expect(readFileSync(fixture.outputFile, 'utf8')).toContain('int8-per-vector-v1');
    expect(statSync(`${fixture.outputFile}.vectors.bin`).size).toBe(9);

    const secondEmbedder = new TestEmbedder();
    const unchanged = await buildSkillsIndex({
      skills: fixture.skills,
      outputFile: fixture.outputFile,
      threshold: 2,
      retrievalLimit: 2,
      embedder: secondEmbedder,
    });
    expect(unchanged.unchanged).toBe(true);
    expect(secondEmbedder.calls).toBe(0);
  });

  test('detects stale metadata while optionally allowing unindexed extra skills', async () => {
    const fixture = await indexedFixture();
    expect(() => assertSkillsIndexCurrent(fixture.index, fixture.skills)).not.toThrow();
    expect(() =>
      assertSkillsIndexCurrent(fixture.index, [
        ...fixture.skills,
        agentSkill({ name: 'extra', description: 'An extra skill.', content: 'extra' }),
      ]),
    ).toThrow(/stale/);
    expect(() =>
      assertSkillsIndexCurrent(
        fixture.index,
        [
          ...fixture.skills,
          agentSkill({ name: 'extra', description: 'An extra skill.', content: 'extra' }),
        ],
        { allowExtraSkills: true },
      ),
    ).not.toThrow();
    expect(() =>
      assertSkillsIndexCurrent(fixture.index, [
        ...fixture.skills.slice(0, 1),
        agentSkill({
          name: 'postgres-reviewer',
          description: 'Changed description.',
          content: 'changed',
        }),
        ...fixture.skills.slice(2),
      ]),
    ).toThrow(/description changed/);
  });
});

describe('skill index retrieval', () => {
  test('uses cosine similarity, top-k, and model identity', async () => {
    const fixture = await indexedFixture();
    expect(cosineSimilarity([1, 0], [0.5, 0.5])).toBeCloseTo(Math.SQRT1_2);
    expect(rankSkillsIndex(fixture.index, [0, 1, 0], { limit: 1 })[0]?.name).toBe(
      'postgres-reviewer',
    );
    const firstEntry = fixture.index.entries[0];
    if (!firstEntry) throw new Error('missing fixture entry');
    expect(scoreSkillsIndexEntry(firstEntry, [1, 0, 0])).toMatchObject({
      score: 1,
      matchedChunk: 0,
      matchedSource: 'document',
    });
    expect(
      scoreSkillsIndexEntry(
        {
          kind: 'skill',
          name: 'chunked',
          description: 'routing frontmatter',
          documentHash: 'test',
          chunks: [
            { source: 'document', embedding: new Float32Array([0, 1]) },
            { source: 'document', embedding: new Float32Array([1, 0]) },
          ],
        },
        [1, 0],
      ),
    ).toMatchObject({ score: 0.25, matchedChunk: 1 });
    expect(
      (
        await searchSkillsIndex(fixture.index, 'extract a PDF table', {
          embedder: fixture.embedder,
          limit: 1,
        })
      )[0]?.name,
    ).toBe('pdf-reader');

    const mismatched: SkillEmbedder = {
      id: 'other@other',
      model: 'other',
      revision: 'other',
      embed: async () => [new Float32Array([1, 0, 0])],
      split: async (text) => [text],
    };
    await expect(searchSkillsIndex(fixture.index, 'pdf', { embedder: mismatched })).rejects.toThrow(
      /query embedder/,
    );
  });

  test('rejects malformed and inconsistent JSONL', () => {
    const directory = mkdtempSync(join(tmpdir(), 'skills-index-bad-'));
    const file = join(directory, 'bad.jsonl');
    writeFileSync(file, '{}\n');
    expect(() => loadSkillsIndex(file)).toThrow(/Unsupported/);
  });

  test('continues to read version-2 JSONL indexes', () => {
    const directory = mkdtempSync(join(tmpdir(), 'skills-index-v2-'));
    const file = join(directory, 'legacy.jsonl');
    const vector = Buffer.alloc(8);
    vector.writeFloatLE(1, 0);
    vector.writeFloatLE(0, 4);
    const metadata = {
      kind: '@di-framework/ai-utils/skills-index',
      version: 2,
      indexed: true,
      skillCount: 1,
      chunkCount: 1,
      threshold: 0,
      retrievalLimit: 1,
      chunkTokens: 256,
      chunkOverlapTokens: 32,
      scoring: 'frontmatter-guided-document-cosine-v1',
      vectorEncoding: 'float32-le-base64',
      catalogHash: 'legacy',
      model: 'fixture',
      revision: '1',
      embedderId: 'fixture@1',
      dimensions: 2,
    };
    const entry = {
      kind: 'skill',
      name: 'legacy',
      description: 'legacy index',
      documentHash: 'hash',
      chunks: [{ source: 'document', vector: vector.toString('base64') }],
    };
    writeFileSync(file, `${JSON.stringify(metadata)}\n${JSON.stringify(entry)}\n`);
    expect(loadSkillsIndex(file).metadata.version).toBe(2);
    expect(rankSkillsIndex(loadSkillsIndex(file), [1, 0])[0]?.name).toBe('legacy');
  });

  test('validates sidecar truncation, hashes, and dimensions', async () => {
    const fixture = await indexedFixture();
    const sidecar = `${fixture.outputFile}.vectors.bin`;
    const original = readFileSync(sidecar);
    writeFileSync(sidecar, original.subarray(0, original.length - 1));
    expect(() => loadSkillsIndex(fixture.outputFile)).toThrow(/truncated/);
    writeFileSync(
      sidecar,
      Buffer.from(original.map((value, index) => (index === 0 ? value ^ 1 : value))),
    );
    expect(() => loadSkillsIndex(fixture.outputFile)).toThrow(/hash mismatch/);
    writeFileSync(sidecar, original);

    const manifest = JSON.parse(readFileSync(fixture.outputFile, 'utf8'));
    manifest.metadata.dimensions += 1;
    writeFileSync(fixture.outputFile, JSON.stringify(manifest));
    expect(() => loadSkillsIndex(fixture.outputFile)).toThrow(/dimension mismatch/);
  });

  test('fuses lexical and dense evidence, pins names, and abstains safely', async () => {
    const fixture = await indexedFixture();
    const rare = await searchSkillsIndex(fixture.index, 'check row-level security policy', {
      embedder: fixture.embedder,
    });
    expect(rare[0]?.name).toBe('postgres-reviewer');
    expect(rare[0]?.lexicalScore).toBeGreaterThan(0);

    const explicit = await searchSkillsIndex(fixture.index, 'please run pdf-reader now', {
      embedder: fixture.embedder,
    });
    expect(explicit[0]).toMatchObject({ name: 'pdf-reader', exactName: true });
    expect(new Set(explicit.map((match) => match.name)).size).toBe(explicit.length);

    expect(
      await searchSkillsIndex(fixture.index, 'compose sonnets about moonlight', {
        embedder: fixture.embedder,
      }),
    ).toEqual([]);
  });
});

describe('SkillsAgent semantic discovery', () => {
  test('default retrieval advisor tool creation supports call and stream chains', async () => {
    const fixture = await indexedFixture();
    const advisor = new SkillsRetrievalAdvisor({
      index: fixture.index,
      skills: fixture.skills,
      embedder: fixture.embedder,
      limit: 2,
    });
    const request = chatClientRequest(
      new Prompt('xpdf-reader then use pdf-reader', {
        toolCallbacks: [skillsTool({ skills: fixture.skills })],
      }),
    );
    const assertSelected = (nextRequest: typeof request) => {
      const description =
        nextRequest.prompt.options?.toolCallbacks?.[0]?.toolDefinition.description;
      expect(description).toContain('<name>pdf-reader</name>');
      expect(nextRequest.context.get('skills_retrieval')).toBeDefined();
    };
    const callChain: CallAdvisorChain = {
      callAdvisors: [],
      async nextCall(nextRequest) {
        assertSelected(nextRequest);
        return chatClientResponse(ChatResponse.of('call'));
      },
    };
    expect((await advisor.adviseCall(request, callChain)).chatResponse?.content).toBe('call');

    const streamChain: StreamAdvisorChain = {
      streamAdvisors: [],
      async *nextStream(nextRequest) {
        assertSelected(nextRequest);
        yield chatClientResponse(ChatResponse.of('stream'));
      },
    };
    const streamed = [];
    for await (const response of advisor.adviseStream(request, streamChain)) {
      streamed.push(response.chatResponse?.content);
    }
    expect(streamed).toEqual(['stream']);
  });

  test('fails clearly when semantic discovery is explicit and the index is missing', () => {
    const workspace = mkdtempSync(join(tmpdir(), 'skills-index-missing-'));
    expect(() =>
      SkillsAgent.builder()
        .chatModel(new ScriptedChatModel([]))
        .addSkills(testSkills())
        .workspace(workspace)
        .semanticDiscovery()
        .build(),
    ).toThrow(/Skills index does not exist/);
  });

  test('shows only top-k skills to the model and can still activate the selected body', async () => {
    const fixture = await indexedFixture();
    const model: ChatModel = new ScriptedChatModel([
      {
        respond: (prompt) => {
          const skillTool = prompt.options?.toolCallbacks?.find(
            (tool) => tool.toolDefinition.name === 'Skill',
          );
          expect(skillTool?.toolDefinition.description).toContain('<name>pdf-reader</name>');
          expect(skillTool?.toolDefinition.description).not.toContain('postgres-reviewer');
          return toolCallResponse([toolCall('skill-1', 'Skill', { command: 'pdf-reader' })]);
        },
      },
      {
        respond: (prompt) => {
          expect(JSON.stringify(prompt.messages)).toContain('Use the PDF parser.');
          return 'done';
        },
      },
    ]);

    const bundle = SkillsAgent.builder()
      .chatModel(model)
      .addSkills(fixture.skills)
      .semanticDiscovery({
        indexFile: fixture.outputFile,
        embedder: fixture.embedder,
        limit: 1,
      })
      .todos(false)
      .buildBundle();

    expect((await bundle.agent.chat('Extract the tables from this PDF.')).content).toBe('done');
    expect(bundle.toolbox.retrievalAdvisor).toBeDefined();
  });

  test('removes the Skill tool when calibrated retrieval abstains', async () => {
    const fixture = await indexedFixture();
    const diagnostics: unknown[] = [];
    const advisor = new SkillsRetrievalAdvisor({
      index: fixture.index,
      skills: fixture.skills,
      embedder: fixture.embedder,
      onDiagnostic: (diagnostic) => diagnostics.push(diagnostic),
    });
    const request = chatClientRequest(
      new Prompt('compose sonnets about moonlight', {
        toolCallbacks: [skillsTool({ skills: fixture.skills })],
      }),
    );
    const next = await advisor.before(request);
    expect(next.prompt.options?.toolCallbacks).toEqual([]);
    expect(next.context.get('skills_retrieval')).toEqual({ decision: 'abstained', matches: [] });
    expect(diagnostics).toMatchObject([
      {
        schema: '@di-framework/skills-retrieval-diagnostic',
        version: 1,
        decision: 'abstained',
        backend: 'local',
        matches: [],
      },
    ]);
  });
});
