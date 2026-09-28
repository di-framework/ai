import { isAiError } from '../model/errors.ts';
import type { A2AAgentExecutor } from './executor.ts';
import type { A2ATaskStore } from './task-store.ts';
import {
  type A2AJsonRpcRequest,
  type A2AJsonRpcResponse,
  A2AMethods,
  type CancelTaskParams,
  createTextMessage,
  type GetTaskParams,
  isTerminalTaskState,
  JsonRpcErrorCodes,
  type ListTasksParams,
  type SendMessageParams,
} from './types.ts';

export interface A2AJsonRpcHandlerOptions {
  readonly taskStore: A2ATaskStore;
  readonly executor?: A2AAgentExecutor;
}

/**
 * Handles A2A JSON-RPC 2.0 requests over HTTP.
 *
 * Implements SendMessage, GetTask, ListTasks, and CancelTask.
 */
export class A2AJsonRpcHandler {
  private readonly taskStore: A2ATaskStore;
  private readonly executor?: A2AAgentExecutor;

  constructor(options: A2AJsonRpcHandlerOptions) {
    this.taskStore = options.taskStore;
    this.executor = options.executor;
  }

  static create(options: A2AJsonRpcHandlerOptions): A2AJsonRpcHandler {
    return new A2AJsonRpcHandler(options);
  }

  async handleRequest(payload: unknown, signal?: AbortSignal): Promise<A2AJsonRpcResponse> {
    if (!payload || typeof payload !== 'object') {
      return {
        jsonrpc: '2.0',
        id: null,
        error: {
          code: JsonRpcErrorCodes.INVALID_REQUEST,
          message: 'Invalid JSON-RPC request body',
        },
      };
    }

    const req = payload as Partial<A2AJsonRpcRequest>;
    const id = req.id ?? null;

    if (req.jsonrpc !== '2.0' || typeof req.method !== 'string') {
      return {
        jsonrpc: '2.0',
        id,
        error: {
          code: JsonRpcErrorCodes.INVALID_REQUEST,
          message: "Request must include jsonrpc: '2.0' and method",
        },
      };
    }

    try {
      switch (req.method) {
        case A2AMethods.SEND_MESSAGE: {
          const result = await this.handleSendMessage(req.params as SendMessageParams, signal);
          return { jsonrpc: '2.0', id, result };
        }
        case A2AMethods.GET_TASK: {
          const result = await this.handleGetTask(req.params as GetTaskParams);
          return { jsonrpc: '2.0', id, result };
        }
        case A2AMethods.LIST_TASKS: {
          const result = await this.handleListTasks(req.params as ListTasksParams);
          return { jsonrpc: '2.0', id, result };
        }
        case A2AMethods.CANCEL_TASK: {
          const result = await this.handleCancelTask(req.params as CancelTaskParams);
          return { jsonrpc: '2.0', id, result };
        }
        default: {
          return {
            jsonrpc: '2.0',
            id,
            error: {
              code: JsonRpcErrorCodes.METHOD_NOT_FOUND,
              message: `Method '${req.method}' not found`,
            },
          };
        }
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Internal RPC error';
      const code = isAiError(err)
        ? JsonRpcErrorCodes.INVALID_PARAMS
        : JsonRpcErrorCodes.INTERNAL_ERROR;
      return {
        jsonrpc: '2.0',
        id,
        error: {
          code,
          message,
        },
      };
    }
  }

  async handleHttpRequest(request: Request): Promise<Response> {
    if (request.method !== 'POST') {
      return new Response(
        JSON.stringify({
          jsonrpc: '2.0',
          id: null,
          error: {
            code: JsonRpcErrorCodes.INVALID_REQUEST,
            message: 'A2A JSON-RPC requests must use HTTP POST',
          },
        }),
        {
          status: 405,
          headers: { 'Content-Type': 'application/json', Allow: 'POST' },
        },
      );
    }

    let body: unknown;
    try {
      body = await request.json();
    } catch (_err) {
      return new Response(
        JSON.stringify({
          jsonrpc: '2.0',
          id: null,
          error: {
            code: JsonRpcErrorCodes.PARSE_ERROR,
            message: 'Parse error: invalid JSON',
          },
        }),
        {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        },
      );
    }

    const response = await this.handleRequest(body, request.signal);
    return new Response(JSON.stringify(response), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  private async handleSendMessage(params: SendMessageParams | undefined, signal?: AbortSignal) {
    if (!params || !params.message) {
      throw new Error("SendMessage requires 'message'");
    }

    let task = params.taskId ? this.taskStore.getTask(params.taskId) : undefined;

    if (params.taskId && !task) {
      throw new Error(`Task '${params.taskId}' not found`);
    }

    if (task && isTerminalTaskState(task.status.state)) {
      throw new Error(
        `Cannot send message to task '${task.id}' in terminal state '${task.status.state}'`,
      );
    }

    if (!task) {
      task = this.taskStore.createTask({
        contextId: params.contextId,
        metadata: params.metadata,
        initialMessage: params.message,
      });
    } else {
      task = this.taskStore.appendMessage(task.id, params.message);
    }

    // Bind abort signal
    if (signal) {
      this.taskStore.bindAbortSignal(task.id, signal);
    }

    // Transition to working state
    task = this.taskStore.updateStatus(task.id, 'working');

    if (this.executor) {
      try {
        const execResult = await this.executor.execute(task, params.message, {
          task,
          signal,
          metadata: params.metadata,
        });

        if (execResult.artifacts) {
          for (const artifact of execResult.artifacts) {
            this.taskStore.appendArtifact(task.id, artifact);
          }
        }

        if (execResult.messages) {
          for (const msg of execResult.messages) {
            this.taskStore.appendMessage(task.id, msg);
          }
        }

        task = this.taskStore.updateStatus(
          task.id,
          execResult.status.state,
          execResult.status.message,
        );
      } catch (err: unknown) {
        if (signal?.aborted) {
          task = this.taskStore.cancelTask(task.id, 'Operation aborted');
        } else {
          const errMsg = err instanceof Error ? err.message : String(err);
          task = this.taskStore.updateStatus(task.id, 'failed', createTextMessage(errMsg, 'agent'));
        }
      }
    }

    const latest = this.taskStore.getTask(task.id);
    return { task: latest };
  }

  private async handleGetTask(params: GetTaskParams | undefined) {
    if (!params || !params.taskId) {
      throw new Error("GetTask requires 'taskId'");
    }

    const task = this.taskStore.getTask(params.taskId, {
      history: params.history,
    });
    if (!task) {
      throw new Error(`Task '${params.taskId}' not found`);
    }

    return { task };
  }

  private async handleListTasks(params: ListTasksParams | undefined) {
    return this.taskStore.listTasks(params ?? {});
  }

  private async handleCancelTask(params: CancelTaskParams | undefined) {
    if (!params || !params.taskId) {
      throw new Error("CancelTask requires 'taskId'");
    }

    const task = this.taskStore.cancelTask(params.taskId, params.reason);
    return { task };
  }
}
