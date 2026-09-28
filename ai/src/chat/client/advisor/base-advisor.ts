import type { ChatClientRequest } from '../chat-client-request.ts';
import type { ChatClientResponse } from '../chat-client-response.ts';
import type {
  CallAdvisor,
  CallAdvisorChain,
  StreamAdvisor,
  StreamAdvisorChain,
} from './advisor.ts';

/**
 * Options for a before/after advisor, aligned with Spring AI {@code BaseAdvisor}.
 */
export interface BeforeAfterAdvisorOptions {
  readonly name: string;
  readonly order: number;
  before?(
    request: ChatClientRequest,
    chain: CallAdvisorChain | StreamAdvisorChain,
  ): ChatClientRequest | Promise<ChatClientRequest>;
  after?(
    response: ChatClientResponse,
    chain: CallAdvisorChain | StreamAdvisorChain,
  ): ChatClientResponse | Promise<ChatClientResponse>;
}

/**
 * Create a call+stream advisor that transforms the request before and/or the
 * response after the rest of the chain runs.
 */
export function createBeforeAfterAdvisor(
  options: BeforeAfterAdvisorOptions,
): CallAdvisor & StreamAdvisor {
  const before = options.before ?? ((req) => req);
  const after = options.after ?? ((res) => res);

  return {
    name: options.name,
    order: options.order,
    async adviseCall(request, chain) {
      const processed = await before(request, chain);
      const response = await chain.nextCall(processed);
      return after(response, chain);
    },
    async *adviseStream(request, chain) {
      const processed = await before(request, chain);
      for await (const response of chain.nextStream(processed)) {
        yield await after(response, chain);
      }
    },
  };
}
