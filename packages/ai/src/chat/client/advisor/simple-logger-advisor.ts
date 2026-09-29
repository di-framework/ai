import type { ChatClientRequest } from '../chat-client-request.ts';
import type { ChatClientResponse } from '../chat-client-response.ts';
import type {
  CallAdvisor,
  CallAdvisorChain,
  StreamAdvisor,
  StreamAdvisorChain,
} from './advisor.ts';

export interface SimpleLoggerAdvisorOptions {
  readonly order?: number;
  readonly logger?: (message: string) => void;
  readonly requestToString?: (request: ChatClientRequest) => string;
  readonly responseToString?: (response: ChatClientResponse) => string;
}

/**
 * Logs request/response around the chain, aligned with Spring AI
 * {@code SimpleLoggerAdvisor}.
 */
export class SimpleLoggerAdvisor implements CallAdvisor, StreamAdvisor {
  readonly name = 'SimpleLoggerAdvisor';
  readonly order: number;
  private readonly log: (message: string) => void;
  private readonly requestToString: (request: ChatClientRequest) => string;
  private readonly responseToString: (response: ChatClientResponse) => string;

  constructor(options: SimpleLoggerAdvisorOptions = {}) {
    this.order = options.order ?? 0;
    this.log = options.logger ?? console.log;
    this.requestToString =
      options.requestToString ??
      ((req) =>
        `request messages=${req.prompt.messages.length} options=${JSON.stringify(req.prompt.options ?? {})}`);
    this.responseToString =
      options.responseToString ??
      ((res) => `response content=${JSON.stringify(res.chatResponse?.content ?? null)}`);
  }

  async adviseCall(
    request: ChatClientRequest,
    chain: CallAdvisorChain,
  ): Promise<ChatClientResponse> {
    this.log(`[SimpleLoggerAdvisor] ${this.requestToString(request)}`);
    const response = await chain.nextCall(request);
    this.log(`[SimpleLoggerAdvisor] ${this.responseToString(response)}`);
    return response;
  }

  async *adviseStream(
    request: ChatClientRequest,
    chain: StreamAdvisorChain,
  ): AsyncIterable<ChatClientResponse> {
    this.log(`[SimpleLoggerAdvisor] stream ${this.requestToString(request)}`);
    for await (const response of chain.nextStream(request)) {
      this.log(`[SimpleLoggerAdvisor] stream ${this.responseToString(response)}`);
      yield response;
    }
  }
}
