import { describe, expect, test } from 'bun:test';
import {
  A2ABus,
  a2aBus,
  ChatClient,
  functionToolCallback,
  PlannerExecutorWorkflow,
  planFingerprint,
  requestContains,
  ScriptedChatModel,
  textResponse,
  toolCall,
  toolCallResponse,
} from '../src/index.ts';

function planJson(partial: {
  goal?: string;
  done: boolean;
  finalAnswer?: string;
  reasoning?: string;
  steps: Array<{
    id: string;
    description: string;
    status: string;
    toolName?: string;
    result?: string;
  }>;
}): string {
  return JSON.stringify({
    goal: partial.goal ?? 'test goal',
    done: partial.done,
    finalAnswer: partial.finalAnswer,
    reasoning: partial.reasoning ?? 'ok',
    steps: partial.steps,
  });
}

describe('PlannerExecutorWorkflow', () => {
  test('plan → act → replan until done', async () => {
    const model = new ScriptedChatModel([
      {
        // initial plan
        respond: planJson({
          done: false,
          steps: [
            {
              id: '1',
              description: 'Look up weather',
              status: 'pending',
              toolName: 'getWeather',
            },
          ],
        }),
      },
      {
        // act with tool
        respond: toolCallResponse([toolCall('c1', 'getWeather', { city: 'Yorktown' })]),
      },
      {
        when: (p) =>
          p.messages.some((m) => (m.text ?? '').includes('Yorktown') || m.messageType === 'tool'),
        respond: textResponse('Weather is 70F'),
      },
      {
        // replan → done
        when: requestContains('Latest observation'),
        respond: planJson({
          done: true,
          finalAnswer: 'It is 70F in Yorktown.',
          steps: [
            {
              id: '1',
              description: 'Look up weather',
              status: 'done',
              result: 'Weather is 70F',
            },
          ],
        }),
      },
    ]);

    const weather = functionToolCallback({
      name: 'getWeather',
      description: 'weather',
      inputSchema: {
        type: 'object',
        properties: { city: { type: 'string' } },
        required: ['city'],
      },
      call: ({ city }: { city: string }) => ({ city, temp: 70 }),
    });

    const pe = PlannerExecutorWorkflow.of(ChatClient.create(model));
    const result = await pe.run('Weather in Yorktown?', {
      tools: [weather],
      maxSteps: 5,
    });

    expect(result.answer).toContain('70F');
    expect(result.plan.done).toBe(true);
    expect(result.rounds.some((r) => r.phase === 'act')).toBe(true);
    expect(result.stepCount).toBeGreaterThanOrEqual(1);
  });

  test('completes when first plan is already done', async () => {
    const model = new ScriptedChatModel([
      {
        respond: planJson({
          done: true,
          finalAnswer: 'Already known.',
          steps: [],
        }),
      },
    ]);
    const result = await new PlannerExecutorWorkflow(ChatClient.create(model)).run('q');
    expect(result.answer).toBe('Already known.');
    expect(result.stepCount).toBe(0);
  });

  test('honors maxSteps', async () => {
    const model = new ScriptedChatModel([
      {
        respond: () =>
          planJson({
            done: false,
            steps: [{ id: '1', description: 'spin', status: 'pending' }],
          }),
      },
      { respond: 'acted' },
      {
        respond: () =>
          planJson({
            done: false,
            // change fingerprint each time via result noise — actually cycle protection
            // might fire first. Keep pending with new ids to avoid cycle but hit maxSteps.
            steps: [
              {
                id: `s-${Math.random()}`,
                description: 'spin again',
                status: 'pending',
              },
            ],
          }),
      },
      { respond: 'acted2' },
      {
        respond: () =>
          planJson({
            done: false,
            steps: [
              {
                id: `s-${Math.random()}`,
                description: 'spin more',
                status: 'pending',
              },
            ],
          }),
      },
      { respond: 'acted3' },
      {
        respond: () =>
          planJson({
            done: false,
            steps: [
              {
                id: `s-${Math.random()}`,
                description: 'never done',
                status: 'pending',
              },
            ],
          }),
      },
    ]);

    await expect(
      PlannerExecutorWorkflow.of(ChatClient.create(model)).run('goal', { maxSteps: 2 }),
    ).rejects.toThrow(/maxSteps/);
  });

  test('cycle protection on repeated plan fingerprint', async () => {
    const sticky = planJson({
      done: false,
      steps: [{ id: '1', description: 'same', status: 'pending' }],
    });
    const model = new ScriptedChatModel([
      { respond: sticky },
      { respond: 'observation' },
      // replan returns identical structure → cycle
      { respond: sticky },
    ]);

    await expect(
      PlannerExecutorWorkflow.of(ChatClient.create(model)).run('goal', { maxSteps: 5 }),
    ).rejects.toThrow(/cycle/);
  });

  test('AbortSignal cancels before plan', async () => {
    const ac = new AbortController();
    ac.abort();
    const model = new ScriptedChatModel([{ respond: 'noop' }]);
    await expect(
      PlannerExecutorWorkflow.of(ChatClient.create(model)).run('g', { signal: ac.signal }),
    ).rejects.toMatchObject({ code: 'cancelled' });
  });

  test('planFingerprint is stable for equal plans', () => {
    const a = {
      goal: 'g',
      done: false,
      steps: [{ id: '1', description: 'd', status: 'pending' as const }],
    };
    const b = {
      goal: 'g',
      done: false,
      reasoning: 'ignored in fingerprint? wait - we ignore only reasoning',
      steps: [{ id: '1', description: 'd', status: 'pending' as const }],
    };
    expect(planFingerprint(a)).toBe(planFingerprint(b));
  });
});

describe('A2ABus', () => {
  test('request/response between registered agents', async () => {
    const bus = A2ABus.create();
    bus.register('researcher', async (msg) => `notes:${msg.content}`);
    bus.register('writer', async (msg, b) => {
      const research = await b.request('writer', 'researcher', msg.content);
      return `Article based on ${research.content}`;
    });

    const reply = await bus.request('user', 'writer', 'graph workflows');
    expect(reply.kind).toBe('response');
    expect(reply.content).toContain('notes:graph workflows');
    expect(reply.content).toContain('Article');
    expect(bus.history.length).toBeGreaterThanOrEqual(3);
  });

  test('human-in-the-loop can rewrite messages', async () => {
    const bus = A2ABus.of({
      onHumanInTheLoop: (msg) => ({
        ...msg,
        content: `approved:${msg.content}`,
      }),
    });
    bus.register('agent', async (msg) => `got:${msg.content}`);
    const reply = await bus.human('user', 'agent', 'please proceed');
    expect(reply.content).toBe('got:approved:please proceed');
  });

  test('human hook can reject', async () => {
    const bus = A2ABus.create({
      onHumanInTheLoop: () => null,
    });
    bus.register('agent', async () => 'ok');
    await expect(bus.human('u', 'agent', 'x')).rejects.toThrow(/human-in-the-loop/);
  });

  test('requireHumanFor gates normal requests', async () => {
    const seen: string[] = [];
    const bus = A2ABus.create({
      requireHumanFor: (m) => m.content.includes('secret'),
      onHumanInTheLoop: (m) => {
        seen.push(m.content);
        return m;
      },
    });
    bus.register('a', async (m) => m.content);
    await bus.request('u', 'a', 'hello');
    await bus.request('u', 'a', 'secret docs');
    expect(seen).toEqual(['secret docs']);
  });

  test('AbortSignal on send', async () => {
    const bus = A2ABus.create();
    const ac = new AbortController();
    ac.abort();
    await expect(bus.send('a', 'b', 'x', { signal: ac.signal })).rejects.toMatchObject({
      code: 'cancelled',
    });
  });

  test('unregister removes a handler so messages fall through unhandled', async () => {
    const bus = A2ABus.create();
    bus.register('agent', async () => 'handled');
    bus.unregister('agent');
    const result = await bus.send('a', 'agent', 'hello');
    expect(result.content).toBe('hello');
  });

  test('inbox() peeks queued messages without invoking a handler', async () => {
    const bus = A2ABus.create();
    await bus.send('a', 'b', 'hello');
    expect(bus.inbox('b').map((m) => m.content)).toEqual(['hello']);
    expect(bus.inbox('unknown-agent')).toEqual([]);
  });

  test('human() always routes through the human hook regardless of requireHumanFor', async () => {
    const seen: string[] = [];
    const bus = A2ABus.create({
      onHumanInTheLoop: (m) => {
        seen.push(m.content);
        return m;
      },
    });
    await bus.human('a', 'b', 'from a human');
    expect(seen).toEqual(['from a human']);
  });

  test('a2aBus() factory creates a working bus', async () => {
    const bus = a2aBus();
    bus.register('agent', async () => 'ok');
    const result = await bus.send('a', 'agent', 'hi');
    expect(result.content).toBe('ok');
  });

  test('inbox trims to maxInboxSize when exceeded', async () => {
    const bus = A2ABus.create({ maxInboxSize: 2 });
    await bus.send('a', 'b', 'one');
    await bus.send('a', 'b', 'two');
    await bus.send('a', 'b', 'three');
    expect(bus.inbox('b').map((m) => m.content)).toEqual(['two', 'three']);
  });
});
