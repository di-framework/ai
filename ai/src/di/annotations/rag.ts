import { AiAnnKeys } from './keys.ts';
import {
  defineMethodAnn,
  defineOnCtor,
  defineParamAnn,
  readMethodAnnMap,
  readOnCtor,
} from './meta.ts';

/** Inject or mark the vector store bean. */
export function VectorStore(
  token: string = 'vectorStore',
): ClassDecorator & ParameterDecorator & PropertyDecorator {
  return qualify(AiAnnKeys.VECTOR_STORE, token);
}

export interface DocumentAnnOptions {
  readonly id?: string;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Mark content/type as a document source. */
export function Document(options: DocumentAnnOptions = {}): ClassDecorator & MethodDecorator {
  return ((target: object, propertyKey?: string | symbol) => {
    if (propertyKey !== undefined) {
      defineMethodAnn(AiAnnKeys.DOCUMENT, target, String(propertyKey), options);
    } else {
      defineOnCtor(AiAnnKeys.DOCUMENT, options, target);
    }
  }) as ClassDecorator & MethodDecorator;
}

export interface IndexedDocumentOptions {
  readonly id?: string;
  readonly metadata?: Readonly<Record<string, unknown>>;
  /** Static text to index; otherwise method return value is used at bootstrap. */
  readonly text?: string;
}

/** Document content to ingest into the vector store at bootstrap. */
export function IndexedDocument(
  textOrOptions: string | IndexedDocumentOptions = {},
): ClassDecorator & MethodDecorator {
  const opts: IndexedDocumentOptions =
    typeof textOrOptions === 'string' ? { text: textOrOptions } : textOrOptions;
  return ((target: object, propertyKey?: string | symbol) => {
    if (propertyKey !== undefined) {
      defineMethodAnn(AiAnnKeys.INDEXED_DOCUMENT, target, String(propertyKey), opts);
    } else {
      defineOnCtor(AiAnnKeys.INDEXED_DOCUMENT, opts, target);
    }
  }) as ClassDecorator & MethodDecorator;
}

/** Mark/inject a document retriever bean. */
export function Retriever(
  token: string = 'documentRetriever',
): ClassDecorator & ParameterDecorator & PropertyDecorator {
  return qualify(AiAnnKeys.RETRIEVER, token);
}

/** Inject or configure chat memory bean. */
export function ChatMemory(
  token: string = 'chatMemory',
): ClassDecorator & ParameterDecorator & PropertyDecorator {
  return qualify(AiAnnKeys.CHAT_MEMORY, token);
}

/** Inject or mark the embedding model bean. */
export function EmbeddingModel(
  token: string = 'embeddingModel',
): ClassDecorator & ParameterDecorator & PropertyDecorator {
  return qualify(AiAnnKeys.EMBEDDING_MODEL, token);
}

function qualify(
  key: string,
  token: string,
): ClassDecorator & ParameterDecorator & PropertyDecorator {
  return ((
    target: object,
    propertyKey?: string | symbol,
    indexOrDescriptor?: number | PropertyDescriptor,
  ) => {
    if (typeof indexOrDescriptor === 'number') {
      defineParamAnn(
        key,
        target,
        propertyKey !== undefined ? String(propertyKey) : undefined,
        indexOrDescriptor,
        token,
      );
      return;
    }
    if (propertyKey !== undefined) {
      defineOnCtor(`${key}:prop:${String(propertyKey)}`, token, target);
      return;
    }
    defineOnCtor(key, token, target);
  }) as ClassDecorator & ParameterDecorator & PropertyDecorator;
}

export function getIndexedDocuments(
  target: object,
): Readonly<Record<string, IndexedDocumentOptions>> {
  return readMethodAnnMap<IndexedDocumentOptions>(AiAnnKeys.INDEXED_DOCUMENT, target);
}

export function getClassIndexedDocument(target: object): IndexedDocumentOptions | undefined {
  return readOnCtor(AiAnnKeys.INDEXED_DOCUMENT, target);
}

export function getVectorStoreToken(target: object): string | undefined {
  return readOnCtor(AiAnnKeys.VECTOR_STORE, target);
}
