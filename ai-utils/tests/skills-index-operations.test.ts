import { describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, unlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  buildSkillsIndex,
  inspectSkillsIndex,
  migrateSkillsIndex,
  querySkillsIndex,
  type SkillEmbedder,
  SkillsIndexOperationError,
  type SkillsIndexOperationProgress,
  validateSkillsIndex,
} from '../src/index.ts';

const embedder: SkillEmbedder = {
  id: 'operations-test@1',
  model: 'operations-test',
  revision: '1',
  async split(text) {
    return [text];
  },
  async embed(texts) {
    return texts.map(() => new Float32Array([1, 0]));
  },
};

async function fixture() {
  const root = mkdtempSync(join(tmpdir(), 'skills-index-operations-'));
  const skillDirectory = join(root, 'skills', 'alpha');
  const outputFile = join(root, 'generated', 'skills.json');
  mkdirSync(skillDirectory, { recursive: true });
  writeFileSync(
    join(skillDirectory, 'SKILL.md'),
    '---\nname: alpha\ndescription: Use for alpha work.\n---\n\nDo alpha work.\n',
  );
  await buildSkillsIndex({
    directories: [join(root, 'skills')],
    outputFile,
    threshold: 0,
    embedder,
  });
  return { root, skillDirectory, outputFile };
}

describe('programmatic skills-index operations', () => {
  test('returns typed inspection data without bodies or vectors', async () => {
    const { outputFile } = await fixture();
    const progress: SkillsIndexOperationProgress[] = [];
    const result = inspectSkillsIndex({
      inputFile: outputFile,
      onProgress: (event) => progress.push(event),
    });

    expect(result).toMatchObject({
      file: outputFile,
      metadata: { version: 3, indexed: true, skillCount: 1 },
      lexicalTerms: expect.any(Number),
      manifestBytes: expect.any(Number),
      vectorBytes: 2,
    });
    expect(JSON.stringify(result)).not.toContain('Do alpha work');
    expect(progress).toEqual([{ operation: 'inspect', phase: 'load', completed: 1, total: 1 }]);
  });

  test('reports source drift as a typed negative result', async () => {
    const { root, skillDirectory, outputFile } = await fixture();
    expect(
      validateSkillsIndex({ inputFile: outputFile, directories: ['skills'], cwd: root }),
    ).toMatchObject({ integrity: 'valid', sourceDrift: null, valid: true, ready: true });

    writeFileSync(join(skillDirectory, 'SKILL.md'), '# Alpha\n\nChanged body.\n');
    const drifted = validateSkillsIndex({
      inputFile: outputFile,
      directories: ['skills'],
      cwd: root,
    });
    expect(drifted.valid).toBe(false);
    expect(drifted.sourceDrift).toContain('stale');
  });

  test('queries through an injected embedder and reports progress', async () => {
    const { outputFile } = await fixture();
    const progress: SkillsIndexOperationProgress[] = [];
    const result = await querySkillsIndex({
      inputFile: outputFile,
      query: 'alpha',
      embedder,
      onProgress: (event) => progress.push(event),
    });

    expect(result).toMatchObject({
      decision: 'selected',
      matches: [{ name: 'alpha', matchedChunk: 0 }],
      timings: {
        loadMs: expect.any(Number),
        embedMs: expect.any(Number),
        searchMs: expect.any(Number),
        totalMs: expect.any(Number),
      },
    });
    expect(progress.map(({ phase }) => phase)).toEqual(['load', 'embed-query', 'search']);
  });

  test('migrates through the package API and reports load/write progress', async () => {
    const { root, outputFile } = await fixture();
    const migratedFile = join(root, 'migrated', 'skills.json');
    const progress: SkillsIndexOperationProgress[] = [];
    const result = migrateSkillsIndex({
      inputFile: outputFile,
      outputFile: migratedFile,
      onProgress: (event) => progress.push(event),
    });

    expect(result).toMatchObject({
      inputFile: outputFile,
      outputFile: migratedFile,
      fromVersion: 3,
      toVersion: 3,
    });
    expect(inspectSkillsIndex({ inputFile: migratedFile }).metadata.version).toBe(3);
    expect(progress.map(({ phase }) => phase)).toEqual(['load', 'write']);
  });

  test('uses stable typed failures for options, sources, indexes, and embeddings', async () => {
    await expect(buildSkillsIndex({ threshold: -1 })).rejects.toMatchObject({
      name: 'SkillsIndexOperationError',
      operation: 'build',
      code: 'INVALID_OPTIONS',
    });
    expect(() => inspectSkillsIndex({ inputFile: 'missing-index.json' })).toThrow(
      SkillsIndexOperationError,
    );
    try {
      inspectSkillsIndex({ inputFile: 'missing-index.json' });
      throw new Error('expected inspect to fail');
    } catch (error) {
      expect(error).toMatchObject({ operation: 'inspect', code: 'INDEX_NOT_FOUND' });
    }

    const { root, outputFile } = await fixture();
    await expect(buildSkillsIndex({ outputFile: root, threshold: 0 })).rejects.toMatchObject({
      operation: 'build',
      code: 'WRITE_FAILED',
    });
    try {
      validateSkillsIndex({ inputFile: outputFile, directories: ['missing-skills'] });
      throw new Error('expected source validation to fail');
    } catch (error) {
      expect(error).toMatchObject({ operation: 'validate', code: 'SOURCE_NOT_FOUND' });
    }
    expect(() =>
      inspectSkillsIndex({
        inputFile: outputFile,
        onProgress() {
          unlinkSync(outputFile);
        },
      }),
    ).toThrow(expect.objectContaining({ operation: 'inspect', code: 'OPERATION_FAILED' }));

    const second = await fixture();
    await expect(
      querySkillsIndex({
        inputFile: second.outputFile,
        query: 'alpha',
        embedder: {
          ...embedder,
          async embed() {
            throw new Error('offline');
          },
        },
      }),
    ).rejects.toMatchObject({ operation: 'query', code: 'EMBEDDING_FAILED' });
    await expect(
      querySkillsIndex({
        inputFile: second.outputFile,
        query: 'alpha',
        embedder: { ...embedder, embed: async () => [new Float32Array([1])] },
      }),
    ).rejects.toMatchObject({ operation: 'query', code: 'EMBEDDING_FAILED' });
    await expect(
      querySkillsIndex({
        inputFile: second.outputFile,
        query: 'alpha',
        embedder,
        limit: 0,
      }),
    ).rejects.toMatchObject({ operation: 'query', code: 'INVALID_OPTIONS' });
    await expect(
      querySkillsIndex({ inputFile: second.outputFile, query: '  ' }),
    ).rejects.toMatchObject({ operation: 'query', code: 'INVALID_OPTIONS' });
    expect(() =>
      migrateSkillsIndex({ inputFile: second.outputFile, outputFile: second.root }),
    ).toThrow(expect.objectContaining({ operation: 'migrate', code: 'WRITE_FAILED' }));
  });
});
