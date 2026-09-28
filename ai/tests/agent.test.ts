import { describe, expect, test } from 'bun:test';
import {
  ChainWorkflow,
  ChatAgent,
  ChatClient,
  callChatContent,
  callChatEntity,
  chainWorkflow,
  chatAgent,
  EvaluatorOptimizerWorkflow,
  evaluatorOptimizerWorkflow,
  extractJsonObject,
  functionToolCallback,
  MessageWindowChatMemory,
  mapPool,
  OrchestratorWorkersWorkflow,
  orchestratorWorkersWorkflow,
  ParallelizationWorkflow,
  PlannerExecutorWorkflow,
  parallelizationWorkflow,
  plannerExecutorWorkflow,
  RoutingWorkflow,
  requestContains,
  routingWorkflow,
  ScriptedChatModel,
  Tool,
  toolCall,
  toolCallResponse,
} from '../src/index.ts';

describe('extractJsonObject', () => {
  test('extracts fenced JSON and rejects an unterminated whitespace-heavy fence', () => {
    expect(extractJsonObject('Result:\n```JSON\n{"ok":true}\n```')).toEqual({ ok: true });
    expect(() => extractJsonObject(`\`\`\`json${' '.repeat(10_000)}`)).toThrow(
      'Could not parse JSON from workflow response',
    );
  });

  test('falls back to brace matching when there is no fence', () => {
    expect(extractJsonObject('here you go: {"ok":true} thanks')).toEqual({ ok: true });
  });

  test('throws when no JSON can be found at all', () => {
    expect(() => extractJsonObject('no json here')).toThrow(
      'Could not parse JSON from workflow response',
    );
  });
});

describe('workflow-utils', () => {
  test('callChatContent forwards a merged signal even without explicit options', async () => {
    const model = new ScriptedChatModel([{ respond: 'ok' }]);
    const client = ChatClient.create(model);
    const controller = new AbortController();
    const content = await callChatContent(client, { user: 'hi', signal: controller.signal });
    expect(content).toBe('ok');
  });

  test('callChatEntity forwards a system prompt and merged options', async () => {
    const model = new ScriptedChatModel([{ respond: '{"name":"Ada"}' }]);
    const client = ChatClient.create(model);
    const entity = await callChatEntity<{ name: string }>(client, {
      user: 'hi',
      system: 'be terse',
      schema: { type: 'object', properties: { name: { type: 'string' } } },
      options: { temperature: 0.2 },
    });
    expect(entity.name).toBe('Ada');
  });

  test('callChatEntity throws AiError when the model returns empty content', async () => {
    const model = new ScriptedChatModel([{ respond: '' }]);
    const client = ChatClient.create(model);
    await expect(
      callChatEntity(client, {
        user: 'hi',
        schema: { type: 'object' },
      }),
    ).rejects.toMatchObject({ code: 'output-validation' });
  });

  test('mapPool runs work with a concurrency cap and preserves order', async () => {
    const results = await mapPool([1, 2, 3, 4], 2, async (n) => n * 10);
    expect(results).toEqual([10, 20, 30, 40]);
  });

  test('mapPool returns an empty array immediately for no items', async () => {
    expect(await mapPool([], 4, async (n) => n)).toEqual([]);
  });

  test('mapPool honors an AbortSignal mid-run', async () => {
    const controller = new AbortController();
    let started = 0;
    const promise = mapPool(
      [1, 2, 3],
      1,
      async (n) => {
        started += 1;
        if (started === 2) controller.abort();
        return n;
      },
      controller.signal,
    );
    await expect(promise).rejects.toBeDefined();
  });
});

describe('ChainWorkflow', () => {
  test('feeds each step output into the next', async () => {
    const model = new ScriptedChatModel([
      {
        when: requestContains('extract'),
        respond: (p) => {
          // first step receives original user text
          expect(p.getContents().toLowerCase()).toContain('alice');
          return 'fact: Alice is 30';
        },
      },
      {
        when: requestContains('summarize'),
        respond: (p) => {
          expect(p.getContents()).toContain('fact: Alice is 30');
          return 'Alice (30)';
        },
      },
    ]);
    const client = ChatClient.create(model);
    const chain = new ChainWorkflow(client, ['extract key facts', 'summarize the facts']);
    const result = await chain.chain('Alice is 30 years old.');
    expect(result).toBe('Alice (30)');
  });

  test('chainDetailed returns intermediate steps', async () => {
    const model = new ScriptedChatModel([{ respond: 'step1' }, { respond: 'step2' }]);
    const detailed = await new ChainWorkflow(ChatClient.create(model), ['A', 'B']).chainDetailed(
      'seed',
    );
    expect(detailed.result).toBe('step2');
    expect(detailed.steps).toHaveLength(2);
    expect(detailed.steps[0]?.output).toBe('step1');
    expect(detailed.steps[1]?.input).toBe('step1');
  });

  test('rejects empty steps', () => {
    expect(() => new ChainWorkflow(ChatClient.create(new ScriptedChatModel([])), [])).toThrow(
      /at least one/,
    );
  });

  test('static of matches constructor factory', async () => {
    const model = new ScriptedChatModel([{ respond: 'ok' }]);
    const client = ChatClient.create(model);
    const viaOf = ChainWorkflow.of(client, ['step']);
    expect(await viaOf.chain('in')).toBe('ok');
  });

  test('chainWorkflow() module factory builds a working instance', async () => {
    const model = new ScriptedChatModel([{ respond: 'ok' }]);
    const viaFactory = chainWorkflow(ChatClient.create(model), ['step']);
    expect(viaFactory).toBeInstanceOf(ChainWorkflow);
    expect(await viaFactory.chain('in')).toBe('ok');
  });
});

class ConcurrentFakeModel {
  readonly calls: string[] = [];
  async call(prompt: {
    messages: readonly { text: string | null; messageType?: string }[];
    options?: { signal?: AbortSignal };
  }) {
    if (prompt.options?.signal?.aborted) {
      throw new Error('cancelled');
    }
    const user =
      [...prompt.messages].reverse().find((m) => m.messageType === 'user' || m.text)?.text ?? '';
    this.calls.push(user);
    await Bun.sleep(5);
    const { ChatResponse } = await import('../src/chat/model/chat-response.ts');
    return ChatResponse.of(`done:${user}`);
  }
}

describe('ParallelizationWorkflow', () => {
  test('maps inputs to ordered results', async () => {
    const concurrent = new ConcurrentFakeModel();
    const model = concurrent as unknown as import('../src/chat/model/chat-model.ts').ChatModel;
    const client = ChatClient.create(model);
    const workflow = new ParallelizationWorkflow(client);
    const results = await workflow.parallel('Analyze stakeholder.', [
      'Customers',
      'Employees',
      'Investors',
    ]);
    expect(results).toEqual(['done:Customers', 'done:Employees', 'done:Investors']);
    expect(concurrent.calls).toHaveLength(3);
  });

  test('static of() / parallelizationWorkflow() factories build a working instance', async () => {
    const model = new ScriptedChatModel([{ respond: 'a' }, { respond: 'b' }]);
    const client = ChatClient.create(model);
    const viaOf = ParallelizationWorkflow.of(client);
    const viaFactory = parallelizationWorkflow(client);
    expect(viaOf).toBeInstanceOf(ParallelizationWorkflow);
    expect(viaFactory).toBeInstanceOf(ParallelizationWorkflow);
    const results = await viaOf.parallel('do', ['x', 'y']);
    expect(results).toEqual(['a', 'b']);
  });
});

describe('RoutingWorkflow', () => {
  test('classifies then invokes route system prompt', async () => {
    const model = new ScriptedChatModel([
      {
        when: requestContains('router'),
        respond: JSON.stringify({
          route: 'billing',
          reasoning: 'double charge',
        }),
      },
      {
        when: requestContains('billing specialist'),
        respond: 'Refund initiated',
      },
    ]);
    const client = ChatClient.create(model);
    const detailed = await new RoutingWorkflow(client).routeDetailed('I was charged twice', {
      billing: 'You are a billing specialist. Help with charges.',
      technical: 'You are a technical support engineer.',
      general: 'You are general support.',
    });
    expect(detailed.route).toBe('billing');
    expect(detailed.result).toBe('Refund initiated');
  });

  test('falls back when route is unknown', async () => {
    const model = new ScriptedChatModel([
      {
        when: requestContains('router'),
        respond: JSON.stringify({ route: 'unknown-route' }),
      },
      {
        respond: 'general help',
      },
    ]);
    const result = await new RoutingWorkflow(ChatClient.create(model)).route(
      'hi',
      {
        billing: 'billing',
        general: 'general system',
      },
      { defaultRoute: 'general' },
    );
    expect(result).toBe('general help');
  });

  test('supports function routes', async () => {
    const model = new ScriptedChatModel([
      {
        respond: JSON.stringify({ route: 'custom' }),
      },
    ]);
    const result = await new RoutingWorkflow(ChatClient.create(model)).route('x', {
      custom: async (input) => `handled:${input}`,
    });
    expect(result).toBe('handled:x');
  });

  test('static of() / routingWorkflow() factories build a working instance', async () => {
    const model = new ScriptedChatModel([
      { respond: JSON.stringify({ route: 'general' }) },
      { respond: 'ok' },
    ]);
    const client = ChatClient.create(model);
    const viaOf = RoutingWorkflow.of(client);
    const viaFactory = routingWorkflow(client);
    expect(viaOf).toBeInstanceOf(RoutingWorkflow);
    expect(viaFactory).toBeInstanceOf(RoutingWorkflow);
    const result = await viaOf.route('x', { general: 'general system' });
    expect(result).toBe('ok');
  });
});

describe('OrchestratorWorkersWorkflow', () => {
  test('plans, runs workers, synthesizes', async () => {
    const model = new ScriptedChatModel([
      {
        when: requestContains('orchestrator'),
        respond: JSON.stringify({
          analysis: 'Need technical and user docs',
          tasks: [
            { type: 'technical', description: 'Write API docs' },
            { type: 'user', description: 'Write user guide' },
          ],
        }),
      },
      {
        when: requestContains('Write API docs'),
        respond: 'TECH DOCS',
      },
      {
        when: requestContains('Write user guide'),
        respond: 'USER DOCS',
      },
      {
        when: requestContains('synthesize'),
        respond: 'COMBINED DOCS',
      },
    ]);

    const result = await new OrchestratorWorkersWorkflow(ChatClient.create(model)).process(
      'Document the REST API',
      {
        // Scripted model is sequential — force concurrency 1
        concurrency: 1,
      },
    );

    expect(result.analysis).toContain('technical');
    expect(result.workerResponses).toHaveLength(2);
    expect(result.workerResponses.map((w) => w.result)).toEqual(['TECH DOCS', 'USER DOCS']);
    expect(result.finalResponse).toBe('COMBINED DOCS');
  });

  test('static of() / orchestratorWorkersWorkflow() factories build a working instance', async () => {
    const model = new ScriptedChatModel([
      {
        respond: JSON.stringify({
          analysis: 'simple task',
          tasks: [{ type: 'solo', description: 'do it' }],
        }),
      },
      { respond: 'WORKER DONE' },
      { respond: 'FINAL' },
    ]);
    const client = ChatClient.create(model);
    const viaOf = OrchestratorWorkersWorkflow.of(client);
    const viaFactory = orchestratorWorkersWorkflow(client);
    expect(viaOf).toBeInstanceOf(OrchestratorWorkersWorkflow);
    expect(viaFactory).toBeInstanceOf(OrchestratorWorkersWorkflow);
    const result = await viaOf.process('task', { concurrency: 1 });
    expect(result.finalResponse).toBe('FINAL');
  });
});

describe('EvaluatorOptimizerWorkflow', () => {
  test('refines until evaluator passes', async () => {
    let gen = 0;
    const model = new ScriptedChatModel([
      {
        when: requestContains('generate'),
        respond: () => {
          gen += 1;
          return gen === 1 ? 'draft v1' : 'draft v2 polished';
        },
      },
      {
        when: requestContains('evaluate'),
        respond: () =>
          gen === 1
            ? JSON.stringify({ pass: false, feedback: 'needs polish' })
            : JSON.stringify({ pass: true, feedback: 'looks good', score: 0.95 }),
      },
      {
        when: requestContains('generate'),
        respond: 'draft v2 polished',
      },
      {
        when: requestContains('evaluate'),
        respond: JSON.stringify({
          pass: true,
          feedback: 'looks good',
          score: 0.95,
        }),
      },
    ]);

    // System prompts contain "generate"/"evaluate"? Our defaults say "You generate" and "You evaluate"
    // requestContains is case-insensitive on message text which includes system+user via getContents?
    // requestContains only checks message texts - system is a message so "You generate" matches generate.

    const refined = await new EvaluatorOptimizerWorkflow(ChatClient.create(model)).loop(
      'Create a counter class',
      { maxIterations: 3 },
    );

    expect(refined.solution).toContain('polished');
    expect(refined.chainOfThought.length).toBeGreaterThanOrEqual(2);
    expect(refined.chainOfThought.at(-1)?.evaluation?.pass).toBe(true);
  });

  test('static of() / evaluatorOptimizerWorkflow() factories build a working instance', async () => {
    const model = new ScriptedChatModel([
      { respond: 'draft' },
      { respond: JSON.stringify({ pass: true, feedback: 'good' }) },
    ]);
    const client = ChatClient.create(model);
    const viaOf = EvaluatorOptimizerWorkflow.of(client);
    const viaFactory = evaluatorOptimizerWorkflow(client);
    expect(viaOf).toBeInstanceOf(EvaluatorOptimizerWorkflow);
    expect(viaFactory).toBeInstanceOf(EvaluatorOptimizerWorkflow);
    const refined = await viaOf.loop('task');
    expect(refined.solution).toBe('draft');
  });
});

describe('ChatAgent', () => {
  test('runs tool-calling loop via ChatClient', async () => {
    const weather = functionToolCallback({
      name: 'getWeather',
      description: 'weather',
      inputSchema: {
        type: 'object',
        properties: { city: { type: 'string' } },
        required: ['city'],
      },
      call: ({ city }: { city: string }) => ({ temp: 68, city }),
    });

    const model = new ScriptedChatModel([
      {
        respond: toolCallResponse([toolCall('c1', 'getWeather', { city: 'Yorktown' })]),
      },
      { respond: '68F in Yorktown' },
    ]);

    const agent = ChatAgent.create({
      chatModel: model,
      system: 'Help with weather.',
      tools: [weather],
    });

    const { content } = await agent.chat('Weather in Yorktown?');
    expect(content).toBe('68F in Yorktown');
  });

  test('uses conversation memory across turns', async () => {
    const memory = MessageWindowChatMemory.builder().maxMessages(20).build();
    const model = new ScriptedChatModel([
      { respond: 'Hello Alice' },
      {
        respond: (p) => {
          // Second turn should include prior history (Alice)
          const texts = p.messages.map((m) => m.text ?? '').join(' ');
          expect(texts).toContain('Alice');
          return 'Your name is Alice';
        },
      },
    ]);

    const agent = ChatAgent.create({
      chatModel: model,
      memory,
      defaultConversationId: 'c1',
    });

    await agent.chat("Hi, I'm Alice");
    const second = await agent.chat('What is my name?');
    expect(second.content).toBe('Your name is Alice');
    expect(second.conversationId).toBe('c1');
  });

  test('fromBuilder() exposes a fluent builder covering every setter', async () => {
    class GreeterTools {
      @Tool({ description: 'says hi' })
      greet({ name }: { name: string }): string {
        return `hi ${name}`;
      }
    }

    const model = new ScriptedChatModel([
      {
        respond: toolCallResponse([toolCall('c1', 'greet', { name: 'Bob' })]),
      },
      { respond: 'greeted Bob' },
    ]);

    const memory = MessageWindowChatMemory.builder().maxMessages(10).build();
    const agent = ChatAgent.fromBuilder(ChatClient.builder(model))
      .system('be nice')
      .advisors()
      .tools(functionToolCallback({ name: 'noop', call: () => 'noop' }))
      .toolBeans(new GreeterTools())
      .memory(memory)
      .defaultConversationId('conv-1')
      .defaultOptions({ temperature: 0.1 })
      .build();

    const { content, conversationId } = await agent.chat('Please greet Bob');
    expect(content).toBe('greeted Bob');
    expect(conversationId).toBe('conv-1');
    expect(agent.client).toBeDefined();
  });

  test('chatAgent() module factory builds a working agent', async () => {
    const model = new ScriptedChatModel([{ respond: 'ok' }]);
    const agent = chatAgent({ chatModel: model });
    const { content } = await agent.chat('hi');
    expect(content).toBe('ok');
  });

  test('chat() honors an AbortSignal and merges options', async () => {
    const model = new ScriptedChatModel([{ respond: 'ok' }]);
    const agent = ChatAgent.create({ chatModel: model });
    const controller = new AbortController();
    const { content } = await agent.chat('hi', {
      signal: controller.signal,
      options: { temperature: 0.5 },
    });
    expect(content).toBe('ok');
  });

  test('chat() throws immediately when the signal is already aborted', async () => {
    const model = new ScriptedChatModel([]);
    const agent = ChatAgent.create({ chatModel: model });
    const controller = new AbortController();
    controller.abort();
    await expect(agent.chat('hi', { signal: controller.signal })).rejects.toThrow();
  });

  test('ChatAgent.create throws when neither chatModel nor chatClient is given', () => {
    expect(() => ChatAgent.create({})).toThrow('ChatAgent requires chatModel or chatClient');
  });
});

describe('PlannerExecutorWorkflow', () => {
  test('plannerExecutorWorkflow() factory builds a workflow equivalent to .of()', () => {
    const client = ChatClient.create(new ScriptedChatModel([]));
    expect(plannerExecutorWorkflow(client)).toBeInstanceOf(PlannerExecutorWorkflow);
    expect(PlannerExecutorWorkflow.of(client)).toBeInstanceOf(PlannerExecutorWorkflow);
  });

  test('run() plans, acts (without tools), replans to done, and returns the final answer', async () => {
    const model = new ScriptedChatModel([
      {
        respond: JSON.stringify({
          goal: 'g',
          done: false,
          steps: [{ id: 's1', description: 'do thing', status: 'pending' }],
        }),
      },
      { respond: 'observation1' },
      {
        respond: JSON.stringify({
          goal: 'g',
          done: true,
          finalAnswer: 'All done',
          steps: [{ id: 's1', description: 'do thing', status: 'done', result: 'observation1' }],
        }),
      },
    ]);
    const workflow = plannerExecutorWorkflow(ChatClient.create(model));
    const result = await workflow.run('g');
    expect(result.answer).toBe('All done');
    expect(result.stepCount).toBe(1);
    expect(result.rounds.map((r) => r.phase)).toEqual(['plan', 'act', 'replan']);
    expect(result.rounds[1]?.action).toEqual({ stepId: 's1', observation: 'observation1' });
  });

  test('run() acts through the ChatClient tools() path when a step has a preferred tool', async () => {
    const model = new ScriptedChatModel([
      {
        respond: JSON.stringify({
          goal: 'g',
          done: false,
          steps: [{ id: 's1', description: 'call tool', toolName: 'noop', status: 'pending' }],
        }),
      },
      { respond: 'tool-observation' },
      {
        respond: JSON.stringify({
          goal: 'g',
          done: true,
          finalAnswer: 'done via tool',
          steps: [
            {
              id: 's1',
              description: 'call tool',
              toolName: 'noop',
              status: 'done',
              result: 'tool-observation',
            },
          ],
        }),
      },
    ]);
    const workflow = plannerExecutorWorkflow(ChatClient.create(model));
    const result = await workflow.run('g', {
      tools: [functionToolCallback({ name: 'noop', call: () => 'noop' })],
    });
    expect(result.answer).toBe('done via tool');
  });

  test('run() forces a replan when no pending steps remain but done is false', async () => {
    const model = new ScriptedChatModel([
      { respond: JSON.stringify({ goal: 'g', done: false, steps: [] }) },
      {
        respond: JSON.stringify({
          goal: 'g',
          done: true,
          finalAnswer: 'finished after replan',
          steps: [],
        }),
      },
    ]);
    const workflow = plannerExecutorWorkflow(ChatClient.create(model));
    const result = await workflow.run('g');
    expect(result.answer).toBe('finished after replan');
    expect(result.rounds.map((r) => r.phase)).toEqual(['plan', 'replan']);
  });

  test('run() falls back to joined step results when finalAnswer is absent', async () => {
    const model = new ScriptedChatModel([
      {
        respond: JSON.stringify({
          goal: 'g',
          done: false,
          steps: [{ id: 's1', description: 'thing', status: 'pending' }],
        }),
      },
      { respond: 'the-observation' },
      {
        respond: JSON.stringify({
          goal: 'g',
          done: true,
          steps: [{ id: 's1', description: 'thing', status: 'done', result: 'the-observation' }],
        }),
      },
    ]);
    const workflow = plannerExecutorWorkflow(ChatClient.create(model));
    const result = await workflow.run('g');
    expect(result.answer).toBe('the-observation');
  });

  test('run() throws synchronously when maxSteps is less than 1', async () => {
    const workflow = plannerExecutorWorkflow(ChatClient.create(new ScriptedChatModel([])));
    await expect(workflow.run('g', { maxSteps: 0 })).rejects.toThrow(/maxSteps must be >= 1/);
  });

  test('run() throws when maxSteps is exceeded without completing the goal', async () => {
    const model = new ScriptedChatModel([
      {
        respond: JSON.stringify({
          goal: 'g',
          done: false,
          steps: [{ id: 's1', description: 'first', status: 'pending' }],
        }),
      },
      { respond: 'obs1' },
      {
        respond: JSON.stringify({
          goal: 'g',
          done: false,
          steps: [{ id: 's2', description: 'second', status: 'pending' }],
        }),
      },
    ]);
    const workflow = plannerExecutorWorkflow(ChatClient.create(model));
    await expect(workflow.run('g', { maxSteps: 1 })).rejects.toThrow(/exceeded maxSteps/);
  });

  test('run() throws when it detects a repeated plan state (cycle protection)', async () => {
    const identicalPlan = { goal: 'g', done: false, steps: [] };
    const model = new ScriptedChatModel([
      { respond: JSON.stringify(identicalPlan) },
      { respond: JSON.stringify(identicalPlan) },
    ]);
    const workflow = plannerExecutorWorkflow(ChatClient.create(model));
    await expect(workflow.run('g')).rejects.toThrow(/repeated plan state/);
  });

  test('run() honors an initialPlan and an already-aborted signal', async () => {
    const workflow = plannerExecutorWorkflow(ChatClient.create(new ScriptedChatModel([])));
    const controller = new AbortController();
    controller.abort();
    await expect(workflow.run('g', { signal: controller.signal })).rejects.toThrow();

    const model = new ScriptedChatModel([
      { respond: JSON.stringify({ goal: 'g', done: true, finalAnswer: 'seeded', steps: [] }) },
    ]);
    const seeded = plannerExecutorWorkflow(ChatClient.create(model));
    const result = await seeded.run('g', {
      initialPlan: { goal: 'g', done: true, finalAnswer: 'from initial plan', steps: [] },
    });
    expect(result.answer).toBe('from initial plan');
    expect(result.rounds).toHaveLength(1);
  });
});
