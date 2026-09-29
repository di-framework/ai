import type { Media } from '../content/media.ts';

/**
 * Document container for text/media content and metadata.
 * Spring AI: {@code Document}.
 *
 * Used in ETL / vector-store / RAG pipelines.
 */
export interface Document {
  readonly id: string;
  /** Text content when this is a text document. */
  readonly text: string | null;
  /** Media content when this is a media document. */
  readonly media: Media | null;
  /**
   * Flat metadata map (string / number / boolean values preferred for vector DBs).
   */
  readonly metadata: Readonly<Record<string, unknown>>;
  /**
   * Relevance / similarity score from retrieval (higher = more similar).
   */
  readonly score: number | null;
  /**
   * Optional precomputed embedding. When set, vector stores store it instead of
   * calling {@code EmbeddingModel.embedDocument}.
   */
  readonly embedding?: ArrayLike<number> | null;
}

export interface DocumentOptions {
  readonly id?: string;
  readonly text?: string | null;
  readonly media?: Media | null;
  readonly metadata?: Readonly<Record<string, unknown>>;
  readonly score?: number | null;
  readonly embedding?: ArrayLike<number> | null;
}

let docIdCounter = 0;

function nextId(): string {
  docIdCounter += 1;
  return `doc-${Date.now().toString(36)}-${docIdCounter.toString(36)}`;
}

/**
 * Create a {@link Document}. Exactly one of {@code text} or {@code media} should be set;
 * if neither is provided, text defaults to empty string.
 */
export function document(options: DocumentOptions = {}): Document {
  const text = options.text !== undefined ? options.text : options.media ? null : '';
  const media = options.media ?? null;
  if (text != null && media != null) {
    throw new Error('exactly one of text or media must be specified');
  }
  if (text == null && media == null) {
    throw new Error('exactly one of text or media must be specified');
  }
  return {
    id: options.id?.trim() ? options.id : nextId(),
    text,
    media,
    metadata: { ...(options.metadata ?? {}) },
    score: options.score ?? null,
    ...(options.embedding !== undefined ? { embedding: options.embedding } : {}),
  };
}

/** Convenience: text document. */
export function textDocument(
  text: string,
  metadata: Readonly<Record<string, unknown>> = {},
  id?: string,
): Document {
  return document({ text, metadata, id });
}

/** Return a copy with a new score (and optional other overrides). */
export function withDocumentScore(
  doc: Document,
  score: number | null,
  overrides: Partial<DocumentOptions> = {},
): Document {
  return document({
    id: overrides.id ?? doc.id,
    text: overrides.text !== undefined ? overrides.text : doc.text,
    media: overrides.media !== undefined ? overrides.media : doc.media,
    metadata: overrides.metadata ?? doc.metadata,
    score,
    embedding: overrides.embedding !== undefined ? overrides.embedding : doc.embedding,
  });
}

export function isTextDocument(doc: Document): boolean {
  return doc.text != null;
}
