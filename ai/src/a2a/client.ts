import { AiError } from '../model/errors.ts';
import { AgentCardHelper } from './agent-card.ts';
import {
  A2A_PROTOCOL_VERSION,
  type A2AArtifact,
  type A2AJsonRpcRequest,
  type A2AJsonRpcResponse,
  type A2AMessage,
  A2AMethods,
  type A2ATask,
  AGENT_CARD_WELL_KNOWN_PATH,
  type AgentCard,
  createTextMessage,
  isTerminalTaskState,
  type ListTasksParams,
  type ListTasksResult,
  type SendMessageResult,
} from './types.ts';

export type A2AFetchLike = (input: string | URL, init?: RequestInit) => Promise<Response>;

export type A2AHeadersInit =
  | Record<string, string>
  | readonly (readonly [string, string])[]
  | Headers
  | Iterable<[string, string]>;

export interface A2AClientOptions {
  /** The origin base URL of the remote A2A agent (e.g. 'https://agent.example.com'). */
  readonly baseUrl: string;
  /** Optional pre-fetched or known Agent Card. */
  readonly card?: AgentCard;
  /** Optional explicit JSON-RPC endpoint URL. */
  readonly rpcUrl?: string;
  /** Custom fetch implementation (defaults to global fetch). */
  readonly fetch?: A2AFetchLike;
  /** Custom static or dynamic HTTP headers (e.g. for authentication). */
  readonly headers?: A2AHeadersInit | (() => A2AHeadersInit | Promise<A2AHeadersInit>);
  /** Default polling interval in ms when waiting for tasks (default 100ms). */
  readonly defaultPollIntervalMs?: number;
  /** Default timeout in ms when waiting for tasks (default 30000ms). */
  readonly defaultTimeoutMs?: number;
}

export interface SendOptions {
  readonly message: string | A2AMessage;
  readonly skill?: string;
  readonly taskId?: string;
  readonly contextId?: string;
  readonly metadata?: Readonly<Record<string, unknown>>;
  readonly signal?: AbortSignal;
}

export interface SendAndWaitOptions extends SendOptions {
  readonly pollIntervalMs?: number;
  readonly timeoutMs?: number;
}

function sanitizeTask(raw: unknown): A2ATask {
  if (!raw || typeof raw !== 'object') {
    throw new AiError('Invalid task object received from A2A server', 'invalid-request');
  }

  const t = raw as Record<string, unknown>;
  const status = (t.status && typeof t.status === 'object' ? t.status : { state: 'submitted' }) as {
    state: import('./types.ts').TaskState;
    message?: A2AMessage;
    timestamp?: string;
  };

  const history: A2AMessage[] = Array.isArray(t.history)
    ? (t.history as A2AMessage[]).map((m) => ({
        role: m.role === 'agent' ? 'agent' : 'user',
        parts: Array.isArray(m.parts) ? m.parts : [],
        ...(m.timestamp ? { timestamp: m.timestamp } : {}),
        ...(m.messageId ? { messageId: m.messageId } : {}),
      }))
    : [];

  const artifacts: A2AArtifact[] = Array.isArray(t.artifacts)
    ? (t.artifacts as A2AArtifact[]).map((a) => ({
        artifactId: String(a.artifactId || 'artifact'),
        ...(a.name ? { name: a.name } : {}),
        ...(a.description ? { description: a.description } : {}),
        ...(a.mimeType ? { mimeType: a.mimeType } : {}),
        ...(a.uri ? { uri: a.uri } : {}),
        ...(Array.isArray(a.parts) ? { parts: a.parts } : {}),
      }))
    : [];

  return {
    id: String(t.id || 'unknown'),
    ...(typeof t.contextId === 'string' ? { contextId: t.contextId } : {}),
    status: {
      state: status.state,
      ...(status.message ? { message: status.message } : {}),
      ...(status.timestamp ? { timestamp: status.timestamp } : {}),
    },
    history,
    artifacts,
    ...(t.metadata && typeof t.metadata === 'object'
      ? { metadata: t.metadata as Record<string, unknown> }
      : {}),
  };
}

/**
 * HTTP-based client for interacting with remote A2A 1.0 agents.
 *
 * Supports discovering agent cards, sending messages, polling tasks,
 * and listing or canceling tasks over HTTP JSON-RPC 2.0.
 */
function trimTrailingSlashes(str: string): string {
  let end = str.length;
  while (end > 0 && str.charCodeAt(end - 1) === 47 /* '/' */) {
    end--;
  }
  return str.slice(0, end);
}

export class A2AClient {
  private readonly baseUrl: string;
  private card?: AgentCard;
  private rpcUrl?: string;
  private readonly customFetch: A2AFetchLike;
  private readonly headers?: A2AHeadersInit | (() => A2AHeadersInit | Promise<A2AHeadersInit>);
  private readonly defaultPollIntervalMs: number;
  private readonly defaultTimeoutMs: number;
  private requestIdCounter = 0;

  private constructor(options: A2AClientOptions) {
    this.baseUrl = trimTrailingSlashes(options.baseUrl);
    this.card = options.card;
    this.rpcUrl = options.rpcUrl;
    this.customFetch = options.fetch ?? globalThis.fetch.bind(globalThis);
    this.headers = options.headers;
    this.defaultPollIntervalMs = options.defaultPollIntervalMs ?? 100;
    this.defaultTimeoutMs = options.defaultTimeoutMs ?? 30_000;
  }

  static create(options: A2AClientOptions): A2AClient {
    return new A2AClient(options);
  }

  static of(baseUrl: string, options?: Omit<A2AClientOptions, 'baseUrl'>): A2AClient {
    return new A2AClient({ baseUrl, ...options });
  }

  private async getResolvedHeaders(): Promise<Record<string, string>> {
    const base: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    };

    if (this.headers) {
      const resolved = typeof this.headers === 'function' ? await this.headers() : this.headers;
      if (resolved instanceof Headers) {
        resolved.forEach((value, key) => {
          base[key] = value;
        });
      } else if (Array.isArray(resolved)) {
        for (const [key, value] of resolved) {
          base[key] = value;
        }
      } else if (resolved && typeof resolved === 'object') {
        Object.assign(base, resolved);
      }
    }

    return base;
  }

  async getCard(forceRefresh = false): Promise<AgentCard> {
    if (this.card && !forceRefresh) {
      return this.card;
    }

    const cardUrl = `${this.baseUrl}${AGENT_CARD_WELL_KNOWN_PATH}`;
    const headers = await this.getResolvedHeaders();

    const response = await this.customFetch(cardUrl, {
      method: 'GET',
      headers,
    });

    if (!response.ok) {
      throw new AiError(
        `Failed to fetch Agent Card from ${cardUrl}: HTTP ${response.status} ${response.statusText}`,
        'invalid-request',
        { status: response.status },
      );
    }

    const rawJson = (await response.json()) as string | object;
    const card = AgentCardHelper.parse(rawJson);

    // Validate protocol version across supported interfaces
    const primaryInterface = card.supported_interfaces?.[0];
    if (primaryInterface && primaryInterface.protocol_version !== A2A_PROTOCOL_VERSION) {
      throw new AiError(
        `Unsupported A2A protocol version '${primaryInterface.protocol_version}'. Expected '${A2A_PROTOCOL_VERSION}'`,
        'invalid-request',
      );
    }

    this.card = card;
    if (primaryInterface?.url) {
      this.rpcUrl = primaryInterface.url.startsWith('http')
        ? primaryInterface.url
        : `${this.baseUrl}${primaryInterface.url.startsWith('/') ? '' : '/'}${primaryInterface.url}`;
    }

    return card;
  }

  async discoverEndpoint(): Promise<string> {
    return this.getRpcEndpoint();
  }

  private async getRpcEndpoint(): Promise<string> {
    if (this.rpcUrl) return this.rpcUrl;
    const card = await this.getCard();
    const iface =
      card.supported_interfaces?.find(
        (i) => i.protocol_binding === 'JSONRPC' || !i.protocol_binding,
      ) ?? card.supported_interfaces?.[0];

    if (iface?.url) {
      this.rpcUrl = iface.url.startsWith('http')
        ? iface.url
        : `${this.baseUrl}${iface.url.startsWith('/') ? '' : '/'}${iface.url}`;
      return this.rpcUrl;
    }

    return this.baseUrl;
  }

  private async rpcCall<TParams, TResult>(
    method: string,
    params?: TParams,
    signal?: AbortSignal,
  ): Promise<TResult> {
    const endpoint = await this.getRpcEndpoint();
    const headers = await this.getResolvedHeaders();
    this.requestIdCounter += 1;
    const requestId = `req-${Date.now()}-${this.requestIdCounter}`;

    const requestBody: A2AJsonRpcRequest<string, TParams> = {
      jsonrpc: '2.0',
      id: requestId,
      method,
      params,
    };

    const response = await this.customFetch(endpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestBody),
      signal,
    });

    if (!response.ok) {
      throw new AiError(
        `A2A JSON-RPC HTTP request failed with status ${response.status}: ${response.statusText}`,
        'invalid-request',
        { status: response.status },
      );
    }

    const json = (await response.json()) as A2AJsonRpcResponse<TResult>;
    if (json.error) {
      throw new AiError(
        `A2A JSON-RPC Error [${json.error.code}]: ${json.error.message}`,
        'invalid-request',
        { cause: json.error },
      );
    }

    return json.result as TResult;
  }

  async send(options: SendOptions): Promise<SendMessageResult> {
    const msg: A2AMessage =
      typeof options.message === 'string'
        ? createTextMessage(options.message, 'user')
        : options.message;

    const result = await this.rpcCall<
      {
        message: A2AMessage;
        skill?: string;
        taskId?: string;
        contextId?: string;
        metadata?: Record<string, unknown>;
      },
      { task?: unknown; message?: A2AMessage }
    >(
      A2AMethods.SEND_MESSAGE,
      {
        message: msg,
        skill: options.skill,
        taskId: options.taskId,
        contextId: options.contextId,
        metadata: options.metadata,
      },
      options.signal,
    );

    return {
      ...(result?.task ? { task: sanitizeTask(result.task) } : {}),
      ...(result?.message ? { message: result.message } : {}),
    };
  }

  async getTask(
    taskId: string,
    options?: { history?: boolean; signal?: AbortSignal },
  ): Promise<A2ATask> {
    const result = await this.rpcCall<
      { taskId: string; history?: boolean },
      { task?: unknown } | unknown
    >(A2AMethods.GET_TASK, { taskId, history: options?.history }, options?.signal);

    const raw =
      result && typeof result === 'object' && 'task' in result
        ? (result as { task: unknown }).task
        : result;

    return sanitizeTask(raw);
  }

  async listTasks(params?: ListTasksParams & { signal?: AbortSignal }): Promise<ListTasksResult> {
    const { signal, ...rest } = params ?? {};
    const result = await this.rpcCall<ListTasksParams, { tasks: unknown[]; nextCursor?: string }>(
      A2AMethods.LIST_TASKS,
      rest,
      signal,
    );

    return {
      tasks: (result?.tasks ?? []).map(sanitizeTask),
      ...(result?.nextCursor ? { nextCursor: result.nextCursor } : {}),
    };
  }

  async cancel(taskId: string, reason?: string, signal?: AbortSignal): Promise<A2ATask> {
    const result = await this.rpcCall<
      { taskId: string; reason?: string },
      { task?: unknown } | unknown
    >(A2AMethods.CANCEL_TASK, { taskId, reason }, signal);

    const raw =
      result && typeof result === 'object' && 'task' in result
        ? (result as { task: unknown }).task
        : result;

    return sanitizeTask(raw);
  }

  async sendAndWait(options: SendAndWaitOptions): Promise<A2ATask> {
    const initial = await this.send(options);
    if (!initial.task) {
      throw new AiError('SendMessage did not return a task object', 'invalid-request');
    }

    let currentTask = initial.task;
    const pollInterval = options.pollIntervalMs ?? this.defaultPollIntervalMs;
    const timeout = options.timeoutMs ?? this.defaultTimeoutMs;
    const startTime = Date.now();

    while (
      !isTerminalTaskState(currentTask.status.state) &&
      currentTask.status.state !== 'input-required' &&
      currentTask.status.state !== 'auth-required'
    ) {
      if (options.signal?.aborted) {
        return this.cancel(currentTask.id, 'Aborted', options.signal);
      }

      if (Date.now() - startTime > timeout) {
        throw new AiError(
          `Timed out waiting for task '${currentTask.id}' to complete after ${timeout}ms`,
          'timeout',
        );
      }

      await new Promise((resolve) => setTimeout(resolve, pollInterval));
      currentTask = await this.getTask(currentTask.id, { signal: options.signal });
    }

    return currentTask;
  }
}
