export type { EmbeddingModel } from './embedding-model.ts';
export { embedDocument } from './embedding-model.ts';
export {
  bagOfWordsEmbedding,
  cosineSimilarity,
  FakeEmbeddingModel,
  type FakeEmbeddingModelOptions,
  l2Normalize,
} from './fake-embedding-model.ts';
export { PrecomputedEmbeddingModel } from './precomputed-embedding-model.ts';
