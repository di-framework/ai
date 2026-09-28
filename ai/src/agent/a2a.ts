import { AiError } from '../model/errors.ts';
import { throwIfAborted } from './workflow-utils.ts';

/**
 * In-process agent-to-agent message kinds.
 * This is a thin local protocol — not a network A2A standard.
 */
export type A2AMessageKind = 'request' | 'response' | 'handoff' | 'human' | 'event';

export interface A2AMessage {
  readonly id: string;
  readonly from: string;
  readonly to: string;
  readonly kind: A2AMessageKind;
  readonly content: string;
  readonly payload?: Readonly<Record<string, unknown>>;
  readonly correlationId?: string;
  readonly createdAt: number;
}

export interface A2ASendOptions {
  readonly signal?: AbortSignal;
  readonly kind?: A2AMessageKind;
  readonly payload?: Readonly<Record<string, unknown>>;
  readonly correlationId?: string;
  readonly id?: string;
}

export type A2AHumanHook = (
  message: A2AMessage,
) => A2AMessage | Promise<A2AMessage | null | undefined> | null | undefined;

export type A2AAgentHandler = (
  message: A2AMessage,
  bus: A2ABus,
) => string | A2AMessage | Promise<string | A2AMessage>;

export interface A2ABusOptions {
  /**
   * Called for messages with kind `human` or when {@link requireHumanFor} matches.
   * Return a replacement message, null to drop, or the same message to continue.
   */
  readonly onHumanInTheLoop?: A2AHumanHook;
  /**
   * If returns true, the outbound message is routed through the human hook first.
   */
  readonly requireHumanFor?: (message: A2AMessage) => boolean;
  /** Max queued messages per agent inbox. Default 100. */
  readonly maxInboxSize?: number;
}

let messageSeq = 0;

function nextId(): string {
  messageSeq += 1;
  return `a2a-${Date.now()}-${messageSeq}`;
}

/**
 * In-process agent-to-agent bus with optional human-in-the-loop hooks.
 *
 * Register handlers with {@link register}, send with {@link send} / {@link request}.
 * No network transport — intentional first cut for multi-agent composition tests.
 *
 * @example
 * ```ts
 * const bus = A2ABus.create();
 * bus.register('researcher', async (msg) => `notes about ${msg.content}`);
 * bus.register('writer', async (msg, b) => {
 *   const research = await b.request('writer', 'researcher', msg.content);
 *   return `Article: ${research.content}`;
 * });
 * const reply = await bus.request('user', 'writer', 'Graph workflows');
 * ```
 */
export class A2ABus {
  private readonly handlers = new Map<string, A2AAgentHandler>();
  private readonly inboxes = new Map<string, A2AMessage[]>();
  private readonly options: A2ABusOptions;
  private readonly maxInbox: number;
  readonly history: A2AMessage[] = [];

  private constructor(options: A2ABusOptions) {
    this.options = options;
    this.maxInbox = options.maxInboxSize ?? 100;
  }

  static create(options: A2ABusOptions = {}): A2ABus {
    return new A2ABus(options);
  }

  static of(options?: A2ABusOptions): A2ABus {
    return A2ABus.create(options);
  }

  register(agentId: string, handler: A2AAgentHandler): this {
    if (!agentId) {
      throw new AiError('A2A agent id is required', 'invalid-request', { retryable: false });
    }
    this.handlers.set(agentId, handler);
    if (!this.inboxes.has(agentId)) {
      this.inboxes.set(agentId, []);
    }
    return this;
  }

  unregister(agentId: string): void {
    this.handlers.delete(agentId);
  }

  /** Peek queued messages for an agent (does not invoke handler). */
  inbox(agentId: string): readonly A2AMessage[] {
    return this.inboxes.get(agentId) ?? [];
  }

  async send(
    from: string,
    to: string,
    content: string,
    options?: A2ASendOptions,
  ): Promise<A2AMessage> {
    throwIfAborted(options?.signal);
    let message: A2AMessage = {
      id: options?.id ?? nextId(),
      from,
      to,
      kind: options?.kind ?? 'request',
      content,
      payload: options?.payload,
      correlationId: options?.correlationId,
      createdAt: Date.now(),
    };

    const needsHuman =
      message.kind === 'human' || (this.options.requireHumanFor?.(message) ?? false);

    if (needsHuman && this.options.onHumanInTheLoop) {
      const maybe = await this.options.onHumanInTheLoop(message);
      if (maybe == null) {
        throw new AiError('A2A human-in-the-loop rejected the message', 'cancelled', {
          retryable: false,
        });
      }
      message = maybe;
    }

    this.history.push(message);
    this.enqueue(to, message);

    const handler = this.handlers.get(to);
    if (handler) {
      const result = await handler(message, this);
      if (typeof result === 'string') {
        const response: A2AMessage = {
          id: nextId(),
          from: to,
          to: from,
          kind: 'response',
          content: result,
          correlationId: message.id,
          createdAt: Date.now(),
        };
        this.history.push(response);
        this.enqueue(from, response);
        return response;
      }
      this.history.push(result);
      this.enqueue(result.to, result);
      return result;
    }

    return message;
  }

  /**
   * Send a request and return the handler response (or the enqueued message when no handler).
   */
  async request(
    from: string,
    to: string,
    content: string,
    options?: A2ASendOptions,
  ): Promise<A2AMessage> {
    return this.send(from, to, content, { ...options, kind: options?.kind ?? 'request' });
  }

  /**
   * Deliver a human message into the bus (always passes through the human hook when set).
   */
  async human(
    from: string,
    to: string,
    content: string,
    options?: Omit<A2ASendOptions, 'kind'>,
  ): Promise<A2AMessage> {
    return this.send(from, to, content, { ...options, kind: 'human' });
  }

  private enqueue(agentId: string, message: A2AMessage): void {
    let box = this.inboxes.get(agentId);
    if (!box) {
      box = [];
      this.inboxes.set(agentId, box);
    }
    box.push(message);
    if (box.length > this.maxInbox) {
      box.splice(0, box.length - this.maxInbox);
    }
  }
}

export function a2aBus(options?: A2ABusOptions): A2ABus {
  return A2ABus.create(options);
}
