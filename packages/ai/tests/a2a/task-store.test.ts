import { describe, expect, it } from 'bun:test';
import { A2ATaskStore, createTextMessage } from '../../src/a2a/index.ts';
import { isAiError } from '../../src/model/errors.ts';

describe('A2ATaskStore', () => {
  it('creates tasks in submitted state and generates ids', () => {
    const store = A2ATaskStore.create();
    const task = store.createTask({
      contextId: 'ctx-123',
      metadata: { workId: 'ticket-456' },
      initialMessage: createTextMessage('Hello Agent', 'user'),
    });

    expect(task.id).toMatch(/^task-/);
    expect(task.contextId).toBe('ctx-123');
    expect(task.status.state).toBe('submitted');
    expect(task.history?.length).toBe(1);
    expect(task.metadata?.workId).toBe('ticket-456');
  });

  it('progresses through valid lifecycle states', () => {
    const store = A2ATaskStore.of();
    const task = store.createTask();

    const working = store.updateStatus(task.id, 'working');
    expect(working.status.state).toBe('working');

    const inputReq = store.updateStatus(
      task.id,
      'input-required',
      createTextMessage('Need details', 'agent'),
    );
    expect(inputReq.status.state).toBe('input-required');
    expect(inputReq.status.message?.parts[0]).toEqual({
      kind: 'text',
      text: 'Need details',
    });

    const workingAgain = store.updateStatus(task.id, 'working');
    expect(workingAgain.status.state).toBe('working');

    const completed = store.updateStatus(task.id, 'completed', createTextMessage('Done', 'agent'));
    expect(completed.status.state).toBe('completed');
  });

  it('rejects illegal state transitions with AiError', () => {
    const store = A2ATaskStore.create();
    const task = store.createTask();
    store.updateStatus(task.id, 'working');
    store.updateStatus(task.id, 'completed');

    expect(() => store.updateStatus(task.id, 'working')).toThrow();
    try {
      store.updateStatus(task.id, 'working');
    } catch (err) {
      expect(isAiError(err)).toBe(true);
      if (isAiError(err)) {
        expect(err.code).toBe('invalid-request');
      }
    }
  });

  it('appends messages and artifacts to a task', () => {
    const store = A2ATaskStore.create();
    const task = store.createTask();

    store.appendMessage(task.id, createTextMessage('Question', 'user'));
    store.appendMessage(task.id, createTextMessage('Answer', 'agent'));

    store.appendArtifact(task.id, {
      artifactId: 'art-1',
      name: 'report.txt',
      parts: [{ kind: 'text', text: 'Report content' }],
    });

    const retrieved = store.getTask(task.id);
    expect(retrieved?.history?.length).toBe(2);
    expect(retrieved?.artifacts?.length).toBe(1);
    expect(retrieved?.artifacts?.[0]?.artifactId).toBe('art-1');
  });

  it('lists tasks with filtering and pagination', () => {
    const store = A2ATaskStore.create();
    const t1 = store.createTask({ contextId: 'ctx-1' });
    const t2 = store.createTask({ contextId: 'ctx-1' });
    const t3 = store.createTask({ contextId: 'ctx-2' });

    store.updateStatus(t1.id, 'working');
    store.updateStatus(t1.id, 'completed');

    const byContext = store.listTasks({ contextId: 'ctx-1' });
    expect(byContext.tasks.length).toBe(2);

    const byState = store.listTasks({ state: 'completed' });
    expect(byState.tasks.length).toBe(1);
    expect(byState.tasks[0]?.id).toBe(t1.id);

    const paged = store.listTasks({ limit: 2 });
    expect(paged.tasks.length).toBe(2);
    expect(paged.nextCursor).toBeDefined();

    const nextPage = store.listTasks({ cursor: paged.nextCursor, limit: 2 });
    expect(nextPage.tasks.length).toBe(1);
    expect(nextPage.tasks[0]?.id).toBe(t3.id);
  });

  it('cancels task on abort signal', () => {
    const store = A2ATaskStore.create();
    const task = store.createTask();
    const controller = new AbortController();

    store.bindAbortSignal(task.id, controller.signal);
    expect(store.getTask(task.id)?.status.state).toBe('submitted');

    controller.abort();
    expect(store.getTask(task.id)?.status.state).toBe('canceled');
  });

  it('bounds retention to maxTasks, supports deleteTask and clear', () => {
    const store = A2ATaskStore.create({ maxTasks: 3 });
    const t1 = store.createTask();
    const t2 = store.createTask();
    const t3 = store.createTask();
    expect(store.size()).toBe(3);

    const t4 = store.createTask();
    expect(store.size()).toBe(3);
    expect(store.getTask(t1.id)).toBeUndefined();
    expect(store.getTask(t4.id)?.id).toBe(t4.id);

    expect(store.deleteTask(t2.id)).toBe(true);
    expect(store.size()).toBe(2);

    store.clear();
    expect(store.size()).toBe(0);
  });
});
