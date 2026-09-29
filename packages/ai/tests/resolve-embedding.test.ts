import { describe, expect, test } from 'bun:test';
import { document, textDocument } from '../src/document/document.ts';
import { FakeEmbeddingModel } from '../src/embedding/fake-embedding-model.ts';
import { PrecomputedEmbeddingModel } from '../src/embedding/precomputed-embedding-model.ts';
import {
  assertFiniteVector,
  resolveDocumentEmbedding,
  resolveQueryEmbedding,
  unitVector,
} from '../src/vectorstore/resolve-embedding.ts';
import { searchRequest } from '../src/vectorstore/search-request.ts';

describe('resolve-embedding', () => {
  test('uses precomputed vectors, normalizes, and rejects invalid values', async () => {
    const model = new PrecomputedEmbeddingModel(2);
    expect(
      await resolveDocumentEmbedding(model, document({ text: 'e', embedding: [3, 4] })),
    ).toEqual([0.6, 0.8]);
    expect(
      await resolveDocumentEmbedding(model, document({ text: 'e', embedding: [1, 0] })),
    ).toEqual([1, 0]);
    expect(
      await resolveDocumentEmbedding(model, document({ text: 'e', embedding: [0, 0] })),
    ).toEqual([0, 0]);
    expect(await resolveQueryEmbedding(model, searchRequest({ queryEmbedding: [0, 1] }))).toEqual([
      0, 1,
    ]);
    await expect(
      resolveQueryEmbedding(model, searchRequest({ query: 'needs embedding' })),
    ).rejects.toThrow(/queryEmbedding/);
    await expect(
      resolveDocumentEmbedding(
        new PrecomputedEmbeddingModel(3),
        document({ text: 'e', embedding: [1, 0] }),
      ),
    ).rejects.toThrow(/dimension mismatch/);
    await expect(
      resolveQueryEmbedding(
        new PrecomputedEmbeddingModel(3),
        searchRequest({ queryEmbedding: [1, 0] }),
      ),
    ).rejects.toThrow(/dimension mismatch/);
    expect(() => assertFiniteVector([1, Number.NaN])).toThrow(/non-finite/);
    expect(unitVector([0, 0])).toEqual([0, 0]);
    expect(() => new PrecomputedEmbeddingModel(2).embedDocument(textDocument('x'))).toThrow(
      /Document.embedding/,
    );
  });

  test('embeds through the model when documents and queries have no precomputed vector', async () => {
    const model = new FakeEmbeddingModel({ dimensions: 4 });
    const fromDocument = await resolveDocumentEmbedding(model, textDocument('hello world'));
    expect(fromDocument).toHaveLength(4);
    const fromQuery = await resolveQueryEmbedding(model, searchRequest({ query: 'hello world' }));
    expect(fromQuery).toEqual(fromDocument);
  });
});
