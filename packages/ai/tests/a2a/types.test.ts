import { describe, expect, it } from 'bun:test';
import {
  A2A_PROTOCOL_VERSION,
  type A2AArtifact,
  type A2AMessage,
  A2AMethods,
  type A2ATask,
  AGENT_CARD_WELL_KNOWN_PATH,
  type AgentCard,
  createTextMessage,
  isTerminalTaskState,
  TERMINAL_TASK_STATES,
} from '../../src/a2a/index.ts';

describe('A2A 1.0 Types', () => {
  it('exposes standard constants', () => {
    expect(A2A_PROTOCOL_VERSION).toBe('1.0');
    expect(AGENT_CARD_WELL_KNOWN_PATH).toBe('/.well-known/agent-card.json');
    expect(A2AMethods.SEND_MESSAGE).toBe('SendMessage');
    expect(A2AMethods.GET_TASK).toBe('GetTask');
    expect(A2AMethods.LIST_TASKS).toBe('ListTasks');
    expect(A2AMethods.CANCEL_TASK).toBe('CancelTask');
  });

  it('identifies terminal task states', () => {
    expect(TERMINAL_TASK_STATES).toContain('completed');
    expect(TERMINAL_TASK_STATES).toContain('failed');
    expect(TERMINAL_TASK_STATES).toContain('canceled');
    expect(TERMINAL_TASK_STATES).toContain('rejected');

    expect(isTerminalTaskState('completed')).toBe(true);
    expect(isTerminalTaskState('failed')).toBe(true);
    expect(isTerminalTaskState('canceled')).toBe(true);
    expect(isTerminalTaskState('rejected')).toBe(true);

    expect(isTerminalTaskState('submitted')).toBe(false);
    expect(isTerminalTaskState('working')).toBe(false);
    expect(isTerminalTaskState('input-required')).toBe(false);
    expect(isTerminalTaskState('auth-required')).toBe(false);
  });

  it('creates helper text messages with ISO timestamp', () => {
    const msg = createTextMessage('Hello agent', 'user', { traceId: '123' });
    expect(msg.role).toBe('user');
    expect(msg.parts).toEqual([{ kind: 'text', text: 'Hello agent' }]);
    expect(msg.metadata).toEqual({ traceId: '123' });
    expect(typeof msg.timestamp).toBe('string');
  });

  it('supports full typed structural definitions for cards, tasks, and artifacts', () => {
    const artifact: A2AArtifact = {
      artifactId: 'art-1',
      name: 'report.txt',
      mimeType: 'text/plain',
      parts: [{ kind: 'text', text: 'Audit report complete' }],
    };

    const task: A2ATask = {
      id: 'task-1',
      contextId: 'ctx-1',
      status: { state: 'completed' },
      artifacts: [artifact],
    };

    const card: AgentCard = {
      name: 'SecurityReviewer',
      description: 'Reviews code security',
      version: '1.0.0',
      supported_interfaces: [
        {
          url: 'https://security.example.com/a2a',
          protocol_version: '1.0',
          protocol_binding: 'JSONRPC',
        },
      ],
      skills: [
        {
          id: 'security.review',
          name: 'Security Review',
          description: 'Perform static security review',
          tags: ['security', 'audit'],
        },
      ],
    };

    expect(task.id).toBe('task-1');
    expect(card.supported_interfaces[0]?.protocol_version).toBe('1.0');
    expect(card.skills[0]?.id).toBe('security.review');
  });
});
