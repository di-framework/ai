import { describe, expect, it } from 'bun:test';
import {
  type A2AAgentExecutor,
  A2AClient,
  AgentCardHelper,
  createA2AHttpHandler,
  createTextMessage,
} from '../../src/a2a/index.ts';

describe('A2A input-required and auth-required resume', () => {
  it('pauses task as input-required and resumes on subsequent SendMessage with taskId', async () => {
    let turn = 0;

    const hitlExecutor: A2AAgentExecutor = {
      async execute(_task, message, _ctx) {
        turn += 1;
        const input = (message.parts[0] as { text: string }).text;

        if (turn === 1) {
          // Pause execution asking for user confirmation
          return {
            status: {
              state: 'input-required',
              message: createTextMessage(
                'Please confirm deployment region (us-east or eu-west)?',
                'agent',
              ),
            },
          };
        }

        // Resumed turn
        return {
          status: {
            state: 'completed',
            message: createTextMessage(`Deployed to region: ${input}`, 'agent'),
          },
          artifacts: [
            {
              artifactId: 'deploy-record',
              name: 'deploy.json',
              parts: [{ kind: 'data', data: { region: input, status: 'ok' } }],
            },
          ],
        };
      },
    };

    const card = AgentCardHelper.create({
      name: 'DeployAgent',
      description: 'Deploys services with approval',
      a2a: { url: 'http://deploy.local/rpc' },
      skills: [{ id: 'infra.deploy', description: 'Deploy infrastructure' }],
    });

    const handler = createA2AHttpHandler({ card, executor: hitlExecutor });

    const client = A2AClient.create({
      baseUrl: 'http://deploy.local',
      fetch: async (input: string | URL, init?: RequestInit) => {
        const urlStr = typeof input === 'string' ? input : input.toString();
        return handler(new Request(urlStr, init));
      },
    });

    // 1. Initial request -> pauses at input-required
    const step1 = await client.send({
      skill: 'infra.deploy',
      message: 'Deploy payment service',
    });

    expect(step1.task).toBeDefined();
    expect(step1.task?.status.state).toBe('input-required');
    expect(step1.task?.status.message?.parts[0]).toEqual({
      kind: 'text',
      text: 'Please confirm deployment region (us-east or eu-west)?',
    });

    // 2. Client sees input-required and sends response with the same taskId
    const taskId = step1.task!.id;
    const step2 = await client.send({
      taskId,
      message: 'us-east',
    });

    expect(step2.task).toBeDefined();
    expect(step2.task?.id).toBe(taskId);
    expect(step2.task?.status.state).toBe('completed');
    expect(step2.task?.artifacts?.length).toBe(1);
    expect(step2.task?.artifacts?.[0]?.artifactId).toBe('deploy-record');
    expect(step2.task?.history?.length).toBe(4);
  });

  it('transitions to failed/canceled when resume is rejected or canceled by client', async () => {
    const rejectingExecutor: A2AAgentExecutor = {
      async execute(_task, message, _ctx) {
        const text = (message.parts[0] as { text: string }).text;
        if (text === 'start') {
          return {
            status: {
              state: 'input-required',
              message: createTextMessage('Confirm action? (yes/no)', 'agent'),
            },
          };
        }
        if (text === 'no') {
          throw new Error('User declined confirmation');
        }
        return { status: { state: 'completed' } };
      },
    };

    const card = AgentCardHelper.create({
      name: 'SafeAgent',
      a2a: { url: 'http://safe.local/rpc' },
      skills: [{ id: 'safe.action', description: 'Safe action' }],
    });

    const handler = createA2AHttpHandler({ card, executor: rejectingExecutor });

    const client = A2AClient.create({
      baseUrl: 'http://safe.local',
      fetch: async (input: string | URL, init?: RequestInit) => {
        const urlStr = typeof input === 'string' ? input : input.toString();
        return handler(new Request(urlStr, init));
      },
    });

    // 1. Initial Send
    const step1 = await client.send({ message: 'start' });
    expect(step1.task?.status.state).toBe('input-required');

    // 2. Reject resume
    const step2 = await client.send({
      taskId: step1.task?.id,
      message: 'no',
    });

    expect(step2.task?.status.state).toBe('failed');
    expect(step2.task?.status.state).not.toBe('working');

    // 3. Or client directly cancels an input-required task
    const step3 = await client.send({ message: 'start' });
    const canceledTask = await client.cancel(step3.task!.id, 'Canceled by user');
    expect(canceledTask.status.state).toBe('canceled');
  });
});
