import { AiError } from '../model/errors.ts';
import {
  type A2AArtifact,
  type A2AMessage,
  type A2ATask,
  isTerminalTaskState,
  type TaskState,
  type TaskStatus,
} from './types.ts';

export interface A2ATaskStoreOptions {
  /** Maximum number of tasks to retain in memory (default 1000). */
  readonly maxTasks?: number;
  /** Optional time-to-live for tasks in milliseconds. */
  readonly ttlMs?: number;
}

export interface CreateTaskOptions {
  readonly id?: string;
  readonly contextId?: string;
  readonly metadata?: Readonly<Record<string, unknown>>;
  readonly initialMessage?: A2AMessage;
}

export interface ListTasksFilter {
  readonly contextId?: string;
  readonly state?: TaskState;
  readonly limit?: number;
  readonly cursor?: string;
}

interface StoredTaskRecord {
  task: A2ATask;
  createdAt: number;
  updatedAt: number;
  abortCleanup?: () => void;
}

const VALID_TRANSITIONS: Readonly<Record<TaskState, readonly TaskState[]>> = {
  submitted: ['working', 'canceled', 'rejected', 'failed'],
  working: ['input-required', 'auth-required', 'completed', 'failed', 'canceled', 'rejected'],
  'input-required': ['working', 'canceled', 'rejected', 'failed'],
  'auth-required': ['working', 'canceled', 'rejected', 'failed'],
  completed: [],
  failed: [],
  canceled: [],
  rejected: [],
};

/**
 * In-memory process-local store for A2A Task lifecycle state.
 *
 * Manages task state transitions, message history, and artifact attachments.
 * Enforces strict 1.0 lifecycle state machine and bounded memory retention.
 */
export class A2ATaskStore {
  private readonly tasks = new Map<string, StoredTaskRecord>();
  private readonly maxTasks: number;
  private readonly ttlMs?: number;
  private taskCounter = 0;

  constructor(options: A2ATaskStoreOptions = {}) {
    this.maxTasks = options.maxTasks ?? 1000;
    this.ttlMs = options.ttlMs;
  }

  static create(options?: A2ATaskStoreOptions): A2ATaskStore {
    return new A2ATaskStore(options);
  }

  static of(options?: A2ATaskStoreOptions): A2ATaskStore {
    return new A2ATaskStore(options);
  }

  private generateTaskId(): string {
    this.taskCounter += 1;
    const rand = Math.random().toString(36).substring(2, 8);
    return `task-${Date.now()}-${this.taskCounter}-${rand}`;
  }

  private pruneExpired(): void {
    if (!this.ttlMs) return;
    const now = Date.now();
    for (const [id, record] of this.tasks.entries()) {
      if (now - record.updatedAt > this.ttlMs) {
        if (record.abortCleanup) record.abortCleanup();
        this.tasks.delete(id);
      }
    }
  }

  private ensureCapacity(): void {
    if (this.tasks.size < this.maxTasks) return;

    // First attempt: evict oldest terminal tasks
    for (const [id, record] of this.tasks.entries()) {
      if (isTerminalTaskState(record.task.status.state)) {
        if (record.abortCleanup) record.abortCleanup();
        this.tasks.delete(id);
        if (this.tasks.size < this.maxTasks) return;
      }
    }

    // Fallback: evict oldest entry
    const oldestKey = this.tasks.keys().next().value;
    if (oldestKey) {
      const record = this.tasks.get(oldestKey);
      if (record?.abortCleanup) record.abortCleanup();
      this.tasks.delete(oldestKey);
    }
  }

  createTask(options: CreateTaskOptions = {}): A2ATask {
    this.pruneExpired();
    this.ensureCapacity();

    const id = options.id ?? this.generateTaskId();
    if (this.tasks.has(id)) {
      throw new AiError(`Task with id '${id}' already exists`, 'invalid-request');
    }

    const nowIso = new Date().toISOString();
    const history: A2AMessage[] = [];
    if (options.initialMessage) {
      history.push(options.initialMessage);
    }

    const task: A2ATask = {
      id,
      ...(options.contextId ? { contextId: options.contextId } : {}),
      status: {
        state: 'submitted',
        timestamp: nowIso,
      },
      history,
      artifacts: [],
      ...(options.metadata ? { metadata: options.metadata } : {}),
    };

    this.tasks.set(id, {
      task,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    });

    return task;
  }

  getTask(taskId: string, options?: { history?: boolean }): A2ATask | undefined {
    this.pruneExpired();
    const record = this.tasks.get(taskId);
    if (!record) return undefined;

    if (options?.history === false) {
      const { history: _, ...rest } = record.task;
      return rest;
    }
    return record.task;
  }

  listTasks(filter: ListTasksFilter = {}): { tasks: readonly A2ATask[]; nextCursor?: string } {
    this.pruneExpired();
    let all = Array.from(this.tasks.values()).map((r) => r.task);

    if (filter.contextId) {
      all = all.filter((t) => t.contextId === filter.contextId);
    }
    if (filter.state) {
      all = all.filter((t) => t.status.state === filter.state);
    }

    let startIndex = 0;
    if (filter.cursor) {
      const idx = all.findIndex((t) => t.id === filter.cursor);
      if (idx >= 0) {
        startIndex = idx + 1;
      }
    }

    const limit = Math.max(1, filter.limit ?? 50);
    const paginated = all.slice(startIndex, startIndex + limit);
    const nextCursor =
      startIndex + limit < all.length ? paginated[paginated.length - 1]?.id : undefined;

    return {
      tasks: paginated,
      ...(nextCursor ? { nextCursor } : {}),
    };
  }

  updateStatus(taskId: string, newState: TaskState, message?: A2AMessage): A2ATask {
    const record = this.tasks.get(taskId);
    if (!record) {
      throw new AiError(`Task '${taskId}' not found`, 'invalid-request');
    }

    const currentState = record.task.status.state;
    if (currentState === newState) {
      return record.task;
    }

    const allowed = VALID_TRANSITIONS[currentState] ?? [];
    if (!allowed.includes(newState)) {
      throw new AiError(
        `Invalid task state transition from '${currentState}' to '${newState}' for task '${taskId}'`,
        'invalid-request',
      );
    }

    const updatedStatus: TaskStatus = {
      state: newState,
      timestamp: new Date().toISOString(),
      ...(message ? { message } : {}),
    };

    const history = [...(record.task.history ?? [])];
    if (message) {
      history.push(message);
    }

    const updatedTask: A2ATask = {
      ...record.task,
      status: updatedStatus,
      history,
    };

    record.task = updatedTask;
    record.updatedAt = Date.now();

    if (isTerminalTaskState(newState) && record.abortCleanup) {
      record.abortCleanup();
      record.abortCleanup = undefined;
    }

    return updatedTask;
  }

  appendMessage(taskId: string, message: A2AMessage): A2ATask {
    const record = this.tasks.get(taskId);
    if (!record) {
      throw new AiError(`Task '${taskId}' not found`, 'invalid-request');
    }

    const history = [...(record.task.history ?? []), message];
    const updatedTask: A2ATask = {
      ...record.task,
      history,
    };

    record.task = updatedTask;
    record.updatedAt = Date.now();
    return updatedTask;
  }

  appendArtifact(taskId: string, artifact: A2AArtifact): A2ATask {
    const record = this.tasks.get(taskId);
    if (!record) {
      throw new AiError(`Task '${taskId}' not found`, 'invalid-request');
    }

    const artifacts = [...(record.task.artifacts ?? []), artifact];
    const updatedTask: A2ATask = {
      ...record.task,
      artifacts,
    };

    record.task = updatedTask;
    record.updatedAt = Date.now();
    return updatedTask;
  }

  cancelTask(taskId: string, reason?: string): A2ATask {
    const record = this.tasks.get(taskId);
    if (!record) {
      throw new AiError(`Task '${taskId}' not found`, 'invalid-request');
    }

    if (isTerminalTaskState(record.task.status.state)) {
      return record.task;
    }

    const message: A2AMessage | undefined = reason
      ? {
          role: 'agent',
          parts: [{ kind: 'text', text: `Task canceled: ${reason}` }],
          timestamp: new Date().toISOString(),
        }
      : undefined;

    return this.updateStatus(taskId, 'canceled', message);
  }

  bindAbortSignal(taskId: string, signal?: AbortSignal): void {
    if (!signal) return;
    const record = this.tasks.get(taskId);
    if (!record) return;

    if (signal.aborted) {
      this.cancelTask(taskId, 'Operation aborted');
      return;
    }

    const onAbort = () => {
      this.cancelTask(taskId, 'Operation aborted');
    };

    signal.addEventListener('abort', onAbort, { once: true });
    record.abortCleanup = () => {
      signal.removeEventListener('abort', onAbort);
    };
  }

  deleteTask(taskId: string): boolean {
    const record = this.tasks.get(taskId);
    if (record?.abortCleanup) {
      record.abortCleanup();
    }
    return this.tasks.delete(taskId);
  }

  size(): number {
    return this.tasks.size;
  }

  clear(): void {
    for (const record of this.tasks.values()) {
      if (record.abortCleanup) record.abortCleanup();
    }
    this.tasks.clear();
  }
}
