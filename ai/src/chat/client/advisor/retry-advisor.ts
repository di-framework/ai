import type { ChatClientRequest } from '../chat-client-request.ts';
import type { ChatClientResponse } from '../chat-client-response.ts';
import type {
  CallAdvisor,
  CallAdvisorChain,
  StreamAdvisor,
  StreamAdvisorChain,
} from './advisor.ts';

export interface RetryAdvisorOptions {
  readonly maxAttempts?: number;
  readonly backoffMs?: number;
  readonly maxBackoffMs?: number;
  readonly jitter?: number;
  readonly shouldRetry?: (error: unknown, attempt: number) => boolean | Promise<boolean>;
  readonly signal?: AbortSignal;
  readonly order?: number;
}

const defaultRetry = (error: unknown): boolean => {
  const details = (error as { details?: { retryable?: boolean } } | null)?.details;
  return details?.retryable === true || error instanceof TypeError;
};

export class RetryAdvisor implements CallAdvisor, StreamAdvisor {
  readonly name = 'RetryAdvisor';
  readonly order: number;
  private readonly options: Required<
    Pick<RetryAdvisorOptions, 'maxAttempts' | 'backoffMs' | 'maxBackoffMs' | 'jitter'>
  > &
    RetryAdvisorOptions;
  constructor(options: RetryAdvisorOptions = {}) {
    this.order = options.order ?? 0;
    this.options = {
      maxAttempts: Math.max(1, options.maxAttempts ?? 3),
      backoffMs: Math.max(0, options.backoffMs ?? 100),
      maxBackoffMs: Math.max(0, options.maxBackoffMs ?? 10_000),
      jitter: Math.max(0, options.jitter ?? 0.2),
      ...options,
    };
  }
  async adviseCall(
    request: ChatClientRequest,
    chain: CallAdvisorChain,
  ): Promise<ChatClientResponse> {
    let attempt = 1;
    for (;;) {
      try {
        return await chain.nextCall(request);
      } catch (error) {
        if (
          attempt >= this.options.maxAttempts ||
          !(await (this.options.shouldRetry ?? defaultRetry)(error, attempt))
        )
          throw error;
        await this.delay(attempt);
        attempt++;
      }
    }
  }
  async *adviseStream(
    request: ChatClientRequest,
    chain: StreamAdvisorChain,
  ): AsyncIterable<ChatClientResponse> {
    let attempt = 1;
    for (;;) {
      try {
        yield* chain.nextStream(request);
        return;
      } catch (error) {
        if (
          attempt >= this.options.maxAttempts ||
          !(await (this.options.shouldRetry ?? defaultRetry)(error, attempt))
        )
          throw error;
        await this.delay(attempt);
        attempt++;
      }
    }
  }
  private async delay(attempt: number): Promise<void> {
    const base = Math.min(this.options.maxBackoffMs, this.options.backoffMs * 2 ** (attempt - 1));
    const factor = this.options.jitter ? 1 + (Math.random() * 2 - 1) * this.options.jitter : 1;
    const ms = Math.max(0, Math.round(base * factor));
    if (!ms) return;
    await new Promise<void>((resolve, reject) => {
      if (this.options.signal?.aborted)
        return reject(this.options.signal.reason ?? new Error('Aborted'));
      const timer = setTimeout(resolve, ms);
      this.options.signal?.addEventListener(
        'abort',
        () => {
          clearTimeout(timer);
          reject(this.options.signal?.reason ?? new Error('Aborted'));
        },
        { once: true },
      );
    });
  }
}

export const retryAdvisor = (options?: RetryAdvisorOptions): RetryAdvisor =>
  new RetryAdvisor(options);
