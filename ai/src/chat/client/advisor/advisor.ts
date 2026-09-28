import type { ChatClientRequest } from '../chat-client-request.ts';
import type { ChatClientResponse } from '../chat-client-response.ts';

/**
 * Parent advisor contract, aligned with Spring AI {@code Advisor}.
 */
export interface Advisor {
  readonly name: string;
  /** Lower values run earlier on the request path. */
  readonly order: number;
}

/**
 * Chain continuation for non-streaming calls.
 * Spring AI: {@code CallAdvisorChain}.
 */
export interface CallAdvisorChain {
  nextCall(request: ChatClientRequest): Promise<ChatClientResponse>;
  readonly callAdvisors: readonly CallAdvisor[];
  /**
   * Restart the chain from the advisor immediately after {@code after}.
   * Used by recursive advisors (e.g. tool-calling loop).
   * Spring AI: {@code CallAdvisorChain.copy}.
   */
  copy?(after: CallAdvisor): CallAdvisorChain;
}

/**
 * Chain continuation for streaming.
 * Spring AI: {@code StreamAdvisorChain}.
 */
export interface StreamAdvisorChain {
  nextStream(request: ChatClientRequest): AsyncIterable<ChatClientResponse>;
  readonly streamAdvisors: readonly StreamAdvisor[];
  /**
   * Restart the stream chain from the advisor immediately after {@code after}.
   * Spring AI: {@code StreamAdvisorChain.copy}.
   */
  copy?(after: StreamAdvisor): StreamAdvisorChain;
}

/**
 * Advisor for synchronous (Promise) chat calls.
 * Spring AI: {@code CallAdvisor}.
 */
export interface CallAdvisor extends Advisor {
  adviseCall(request: ChatClientRequest, chain: CallAdvisorChain): Promise<ChatClientResponse>;
}

/**
 * Advisor for streaming chat calls.
 * Spring AI: {@code StreamAdvisor}.
 */
export interface StreamAdvisor extends Advisor {
  adviseStream(
    request: ChatClientRequest,
    chain: StreamAdvisorChain,
  ): AsyncIterable<ChatClientResponse>;
}

/**
 * Advisor that participates in both call and stream paths.
 * Spring AI: {@code BaseAdvisor} (before/after helpers available separately).
 */
export type AroundAdvisor = CallAdvisor & StreamAdvisor;

export function isCallAdvisor(advisor: Advisor): advisor is CallAdvisor {
  return typeof (advisor as CallAdvisor).adviseCall === 'function';
}

export function isStreamAdvisor(advisor: Advisor): advisor is StreamAdvisor {
  return typeof (advisor as StreamAdvisor).adviseStream === 'function';
}
