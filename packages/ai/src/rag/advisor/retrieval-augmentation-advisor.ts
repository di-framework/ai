import type {
  CallAdvisor,
  CallAdvisorChain,
  StreamAdvisor,
  StreamAdvisorChain,
} from '../../chat/client/advisor/advisor.ts';
import {
  type ChatClientRequest,
  copyChatClientRequest,
} from '../../chat/client/chat-client-request.ts';
import {
  type ChatClientResponse,
  copyChatClientResponse,
} from '../../chat/client/chat-client-response.ts';
import type { ChatMessage } from '../../chat/messages/message.ts';
import { chatResponseMetadata } from '../../chat/metadata/chat-response-metadata.ts';
import { ChatResponse } from '../../chat/model/chat-response.ts';
import type { Document } from '../../document/document.ts';
import { ContextualQueryAugmenter } from '../generation/contextual-query-augmenter.ts';
import type { QueryAugmenter } from '../generation/query-augmenter.ts';
import { type Query, query } from '../query.ts';
import { ConcatenationDocumentJoiner, type DocumentJoiner } from '../retrieval/document-joiner.ts';
import type { DocumentRetriever } from '../retrieval/document-retriever.ts';

/**
 * Context key for documents retrieved during RAG.
 * Spring AI: {@code RetrievalAugmentationAdvisor.DOCUMENT_CONTEXT}.
 */
export const RAG_DOCUMENT_CONTEXT = 'rag_document_context';

export interface RetrievalAugmentationAdvisorOptions {
  readonly documentRetriever: DocumentRetriever;
  readonly queryAugmenter?: QueryAugmenter;
  readonly documentJoiner?: DocumentJoiner;
  /**
   * Optional query transformers applied in order before retrieval.
   * Spring AI: {@code QueryTransformer}.
   */
  readonly queryTransformers?: readonly QueryTransformer[];
  /**
   * Optional expander that turns one query into many.
   * Spring AI: {@code QueryExpander}.
   */
  readonly queryExpander?: QueryExpander;
  /**
   * Post-processors applied after join, before augmentation.
   * Spring AI: {@code DocumentPostProcessor}.
   */
  readonly documentPostProcessors?: readonly DocumentPostProcessor[];
  /** Advisor order. Default 0 (Spring AI). */
  readonly order?: number;
}

/** Spring AI: {@code QueryTransformer}. */
export interface QueryTransformer {
  transform(query: Query): Query | Promise<Query>;
}

/** Spring AI: {@code QueryExpander}. */
export interface QueryExpander {
  expand(query: Query): readonly Query[] | Promise<readonly Query[]>;
}

/** Spring AI: {@code DocumentPostProcessor}. */
export interface DocumentPostProcessor {
  process(
    query: Query,
    documents: readonly Document[],
  ): readonly Document[] | Promise<readonly Document[]>;
}

/**
 * Modular RAG advisor: transform → expand → retrieve → join → post-process → augment.
 * Spring AI: {@code RetrievalAugmentationAdvisor}.
 */
export class RetrievalAugmentationAdvisor implements CallAdvisor, StreamAdvisor {
  readonly name = 'Retrieval Augmentation Advisor';
  readonly order: number;

  private readonly documentRetriever: DocumentRetriever;
  private readonly queryAugmenter: QueryAugmenter;
  private readonly documentJoiner: DocumentJoiner;
  private readonly queryTransformers: readonly QueryTransformer[];
  private readonly queryExpander: QueryExpander | undefined;
  private readonly documentPostProcessors: readonly DocumentPostProcessor[];

  constructor(options: RetrievalAugmentationAdvisorOptions) {
    if (options.documentRetriever == null) {
      throw new Error('documentRetriever cannot be null');
    }
    this.documentRetriever = options.documentRetriever;
    this.queryAugmenter = options.queryAugmenter ?? ContextualQueryAugmenter.builder();
    this.documentJoiner = options.documentJoiner ?? new ConcatenationDocumentJoiner();
    this.queryTransformers = options.queryTransformers ?? [];
    this.queryExpander = options.queryExpander;
    this.documentPostProcessors = options.documentPostProcessors ?? [];
    this.order = options.order ?? 0;
  }

  static builder(options: RetrievalAugmentationAdvisorOptions): RetrievalAugmentationAdvisor {
    return new RetrievalAugmentationAdvisor(options);
  }

  async before(request: ChatClientRequest): Promise<ChatClientRequest> {
    const contextRecord = mapToRecord(request.context);
    const userText = request.prompt.getUserMessage().text ?? '';

    const originalQuery = query({
      text: userText,
      history: request.prompt.messages as ChatMessage[],
      context: contextRecord,
    });

    let transformedQuery = originalQuery;
    for (const transformer of this.queryTransformers) {
      transformedQuery = await transformer.transform(transformedQuery);
    }

    const expandedQueries = this.queryExpander
      ? await this.queryExpander.expand(transformedQuery)
      : [transformedQuery];

    const documentsForQuery = new Map<Query, readonly (readonly Document[])[]>();
    for (const q of expandedQueries) {
      const docs = await this.documentRetriever.retrieve(q);
      documentsForQuery.set(q, [docs]);
    }

    let documents = this.documentJoiner.join(documentsForQuery);

    for (const post of this.documentPostProcessors) {
      documents = await post.process(originalQuery, documents);
    }

    // Mutable context for chain
    request.context.set(RAG_DOCUMENT_CONTEXT, documents);

    const augmentedQuery = this.queryAugmenter.augment(originalQuery, documents);

    return copyChatClientRequest(request, {
      prompt: request.prompt.augmentUserMessage(augmentedQuery.text),
      context: request.context,
    });
  }

  after(response: ChatClientResponse): ChatClientResponse {
    const docs = response.context.get(RAG_DOCUMENT_CONTEXT);
    if (docs == null) {
      return response;
    }
    const chatResponse = response.chatResponse;
    if (!chatResponse) {
      return response;
    }
    const metadata = chatResponseMetadata({
      ...chatResponse.metadata,
      [RAG_DOCUMENT_CONTEXT]: docs,
    });
    return copyChatClientResponse(response, {
      chatResponse: new ChatResponse(chatResponse.generations, metadata),
    });
  }

  async adviseCall(
    request: ChatClientRequest,
    chain: CallAdvisorChain,
  ): Promise<ChatClientResponse> {
    const processed = await this.before(request);
    const response = await chain.nextCall(processed);
    return this.after(response);
  }

  async *adviseStream(
    request: ChatClientRequest,
    chain: StreamAdvisorChain,
  ): AsyncIterable<ChatClientResponse> {
    const processed = await this.before(request);
    for await (const response of chain.nextStream(processed)) {
      yield this.after(response);
    }
  }
}

function mapToRecord(map: Map<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of map) {
    out[k] = v;
  }
  return out;
}
