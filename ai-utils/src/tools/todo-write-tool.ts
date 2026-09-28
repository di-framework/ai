import { functionToolCallback, type ToolCallback } from '@di-framework/ai';

export type TodoStatus = 'pending' | 'in_progress' | 'completed';

export interface TodoItem {
  readonly content: string;
  readonly activeForm?: string;
  readonly status: TodoStatus;
}

export interface TodoWriteInput {
  readonly todos?: readonly TodoItem[];
}

export interface TodoWriteToolOptions {
  readonly onChange?: (todos: readonly TodoItem[]) => void;
}

export function todoWriteTool(options: TodoWriteToolOptions = {}): ToolCallback {
  let current: TodoItem[] = [];

  return functionToolCallback<TodoWriteInput, string>({
    name: 'TodoWrite',
    description: `Update the structured task list for this session.

Usage:
- Use for multi-step work (3+ steps)
- Exactly one task should be in_progress
- Mark completed only when the step is fully done
- Send the full list on every update`,
    inputSchema: {
      type: 'object',
      properties: {
        todos: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              content: { type: 'string', description: 'Imperative task text' },
              activeForm: { type: 'string', description: 'Present-continuous form' },
              status: { type: 'string', enum: ['pending', 'in_progress', 'completed'] },
            },
            required: ['content', 'status'],
          },
        },
      },
      required: ['todos'],
    },
    call: (input) => {
      const todos = [...(input?.todos ?? [])];
      if (todos.length === 0) return 'Error: todos must not be empty';
      const inProgress = todos.filter((todo) => todo.status === 'in_progress');
      if (inProgress.length > 1) {
        return 'Error: exactly one task may be in_progress';
      }
      current = todos;
      options.onChange?.(current);
      const done = todos.filter((todo) => todo.status === 'completed').length;
      const lines = todos.map((todo) => {
        const mark = todo.status === 'completed' ? '✓' : todo.status === 'in_progress' ? '→' : ' ';
        return `[${mark}] ${todo.content}`;
      });
      const pct = Math.round((done / todos.length) * 100);
      return `Progress: ${done}/${todos.length} tasks completed (${pct}%)\n${lines.join('\n')}`;
    },
  });
}
