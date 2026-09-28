import { describe, expect, it } from 'bun:test';
import {
  type A2AAgentExecutor,
  A2AJsonRpcHandler,
  A2ATaskStore,
  createTextMessage,
  JsonRpcErrorCodes,
} from '../../src/a2a/index.ts';

describe('A2A JSON-RPC Operations', () => {
  it('handles SendMessage and creates a working/completed task via executor', async () => {
    const taskStore = A2ATaskStore.create();

    const mockExecutor: A2AAgentExecutor = {
      async execute(task, message, _ctx) {
        return {
          status: { state: 'completed' },
          messages: [
            createTextMessage(`Processed: ${(message.parts[0] as { text: string }).text}`, 'agent'),
          ],
          artifacts: [
            {
              artifactId: 'art-1',
              name: 'out.txt',
              parts: [{ kind: 'text', text: 'Result artifact' }],
            },
          ],
        };
      },
    };

    const handler = A2AJsonRpcHandler.create({
      taskStore,
      executor: mockExecutor,
    });

    const response = await handler.handleRequest({
      jsonrpc: '2.0',
      id: 'req-1',
      method: 'SendMessage',
      params: {
        message: createTextMessage('Analyze dataset'),
        contextId: 'ctx-1',
      },
    });

    expect(response.error).toBeUndefined();
    expect(response.id).toBe('req-1');
    const result = response.result as {
      task: { id: string; status: { state: string }; artifacts: unknown[]; history: unknown[] };
    };
    expect(result.task.status.state).toBe('completed');
    expect(result.task.artifacts.length).toBe(1);
    expect(result.task.history.length).toBe(2);
  });

  it('handles GetTask, ListTasks, and CancelTask via JSON-RPC HTTP requests', async () => {
    const taskStore = A2ATaskStore.create();
    const task = taskStore.createTask({
      id: 'task-100',
      contextId: 'ctx-100',
      initialMessage: createTextMessage('Initial msg'),
    });

    const handler = A2AJsonRpcHandler.create({ taskStore });

    // HTTP POST GetTask
    const getReq = new Request('http://localhost/rpc', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: '2',
        method: 'GetTask',
        params: { taskId: 'task-100' },
      }),
    });
    const getRes = await handler.handleHttpRequest(getReq);
    expect(getRes.status).toBe(200);
    const getJson = (await getRes.json()) as { result: { task: { id: string } } };
    expect(getJson.result.task.id).toBe('task-100');

    // HTTP POST ListTasks
    const listReq = new Request('http://localhost/rpc', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: '3',
        method: 'ListTasks',
        params: { contextId: 'ctx-100' },
      }),
    });
    const listRes = await handler.handleHttpRequest(listReq);
    const listJson = (await listRes.json()) as { result: { tasks: unknown[] } };
    expect(listJson.result.tasks.length).toBe(1);

    // HTTP POST CancelTask
    const cancelReq = new Request('http://localhost/rpc', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: '4',
        method: 'CancelTask',
        params: { taskId: 'task-100', reason: 'User requested cancellation' },
      }),
    });
    const cancelRes = await handler.handleHttpRequest(cancelReq);
    const cancelJson = (await cancelRes.json()) as {
      result: { task: { status: { state: string } } };
    };
    expect(cancelJson.result.task.status.state).toBe('canceled');
  });

  it('fails closed with method not found error on unknown methods', async () => {
    const taskStore = A2ATaskStore.create();
    const handler = A2AJsonRpcHandler.create({ taskStore });

    const response = await handler.handleRequest({
      jsonrpc: '2.0',
      id: '5',
      method: 'UnknownMethod',
    });

    expect(response.error).toBeDefined();
    expect(response.error?.code).toBe(JsonRpcErrorCodes.METHOD_NOT_FOUND);
  });

  it('supports long-running working tasks without immediate completion', async () => {
    const taskStore = A2ATaskStore.create();
    // Executor that keeps task in working state
    const workingExecutor: A2AAgentExecutor = {
      async execute(_task, _message, _ctx) {
        return {
          status: { state: 'working' },
        };
      },
    };

    const handler = A2AJsonRpcHandler.create({
      taskStore,
      executor: workingExecutor,
    });

    const response = await handler.handleRequest({
      jsonrpc: '2.0',
      id: '6',
      method: 'SendMessage',
      params: { message: createTextMessage('Long running work') },
    });

    const result = response.result as { task: { status: { state: string } } };
    expect(result.task.status.state).toBe('working');
  });
});
