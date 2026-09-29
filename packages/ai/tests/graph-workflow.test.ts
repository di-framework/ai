import { describe, expect, test } from 'bun:test';
import {
  ChatClient,
  chatToolLoopGraph,
  functionToolCallback,
  GRAPH_FINISH,
  GRAPH_START,
  GraphWorkflow,
  graphWorkflow,
  isAiError,
  ScriptedChatModel,
  simpleAgentGraph,
  textResponse,
  toolCall,
  toolCallResponse,
} from '../src/index.ts';

describe('GraphWorkflow', () => {
  test('linear flow carries values and path metadata', async () => {
    const graph = GraphWorkflow.builder<number, string>('linear')
      .node('double', (n: number) => n * 2)
      .node('label', (n: number) => `value=${n}`)
      .edge(GRAPH_START, 'double')
      .edge('double', 'label')
      .edge('label', GRAPH_FINISH)
      .build();

    const result = await graph.run(21);
    expect(result.output).toBe('value=42');
    expect(result.path).toEqual([GRAPH_START, 'double', 'label', GRAPH_FINISH]);
    expect(result.stepCount).toBe(4);
    expect(result.steps.map((s) => s.nodeId)).toEqual([
      GRAPH_START,
      'double',
      'label',
      GRAPH_FINISH,
    ]);
  });

  test('conditional branch selects first matching edge', async () => {
    const graph = GraphWorkflow.builder<{ n: number }, string>('branch')
      .node('classify', (input: { n: number }) => input)
      .node('even', () => 'even')
      .node('odd', () => 'odd')
      .edge(GRAPH_START, 'classify')
      .edge('classify', 'even', { when: (v: { n: number }) => v.n % 2 === 0 })
      .edge('classify', 'odd', { when: (v: { n: number }) => v.n % 2 !== 0 })
      .edge('even', GRAPH_FINISH)
      .edge('odd', GRAPH_FINISH)
      .build();

    expect((await graph.run({ n: 4 })).output).toBe('even');
    expect((await graph.run({ n: 5 })).output).toBe('odd');
  });

  test('async edge transform rewrites values', async () => {
    const graph = GraphWorkflow.builder<string, number>('transform')
      .node('echo', async (s: string) => s)
      .edge(GRAPH_START, 'echo')
      .edge('echo', GRAPH_FINISH, {
        transform: async (s: string) => s.length,
        name: 'to-length',
      })
      .build();

    const result = await graph.run('abcd');
    expect(result.output).toBe(4);
    expect(result.steps.at(-1)?.edgeName).toBe('to-length');
  });

  test('loop with state and finish condition', async () => {
    const graph = GraphWorkflow.builder<number, number>('loop')
      .node('inc', (n: number, ctx) => {
        const count = (Number(ctx.state.rounds ?? 0) + 1) as number;
        ctx.state.rounds = count;
        return n + 1;
      })
      .edge(GRAPH_START, 'inc')
      .edge('inc', 'inc', {
        when: (n: number) => n < 5,
        name: 'again',
      })
      .edge('inc', GRAPH_FINISH, {
        when: (n: number) => n >= 5,
        name: 'done',
      })
      .build();

    const result = await graph.run(0);
    expect(result.output).toBe(5);
    expect(result.path.filter((p) => p === 'inc')).toHaveLength(5);
  });

  test('nested subgraph', async () => {
    const inner = GraphWorkflow.builder<number, number>('inner')
      .node('times10', (n: number) => n * 10)
      .edge(GRAPH_START, 'times10')
      .edge('times10', GRAPH_FINISH)
      .build();

    const outer = GraphWorkflow.builder<number, string>('outer')
      .node('plus1', (n: number) => n + 1)
      .subgraph('scale', inner)
      .node('fmt', (n: number) => `n=${n}`)
      .edge(GRAPH_START, 'plus1')
      .edge('plus1', 'scale')
      .edge('scale', 'fmt')
      .edge('fmt', GRAPH_FINISH)
      .build();

    const result = await outer.run(3);
    expect(result.output).toBe('n=40');
    expect(result.path.some((p) => p.includes('scale/'))).toBe(true);
  });

  test('validation rejects finish with outgoing edges', () => {
    expect(() =>
      GraphWorkflow.builder('bad')
        .node('a', () => 1)
        .edge(GRAPH_START, 'a')
        .edge('a', GRAPH_FINISH)
        .edge(GRAPH_FINISH, 'a')
        .build(),
    ).toThrow(/finish/);
  });

  test('validation rejects unreachable finish', () => {
    expect(() =>
      GraphWorkflow.builder('bad')
        .node('a', () => 1)
        .node('b', () => 2)
        .edge(GRAPH_START, 'a')
        .edge('b', GRAPH_FINISH)
        .build(),
    ).toThrow(/finish is not reachable|no outgoing/);
  });

  test('validation rejects unknown edge target', () => {
    expect(() =>
      GraphWorkflow.builder('bad')
        .node('a', () => 1)
        .edge(GRAPH_START, 'a')
        .edge('a', 'missing')
        .build(),
    ).toThrow(/unknown node/);
  });

  test('validation rejects duplicate node ids', () => {
    expect(() =>
      GraphWorkflow.builder('bad')
        .node('a', () => 1)
        .node('a', () => 2)
        .edge(GRAPH_START, 'a')
        .edge('a', GRAPH_FINISH)
        .build(),
    ).toThrow(/duplicate/);
  });

  test('fails when no edge matches', async () => {
    const graph = GraphWorkflow.builder<number, number>('no-match')
      .node('x', (n: number) => n)
      .edge(GRAPH_START, 'x')
      .edge('x', GRAPH_FINISH, { when: () => false })
      .build();

    await expect(graph.run(1)).rejects.toThrow(/no matching edge/);
  });

  test('honors maxSteps', async () => {
    // Structural finish edge is always present for validation; runtime predicate never matches.
    const looping = GraphWorkflow.builder<number, number>('steps')
      .node('inc', (n: number) => n + 1)
      .edge(GRAPH_START, 'inc')
      .edge('inc', 'inc', { when: async () => true, name: 'loop' })
      .edge('inc', GRAPH_FINISH, { when: async () => false, name: 'never' })
      .build();

    await expect(looping.run(0, { maxSteps: 5 })).rejects.toThrow(/maxSteps/);
  });

  test('honors AbortSignal', async () => {
    const ac = new AbortController();
    const graph = GraphWorkflow.builder<number, number>('cancel')
      .node('slow', async (n: number) => {
        ac.abort();
        return n;
      })
      .node('next', (n: number) => n + 1)
      .edge(GRAPH_START, 'slow')
      .edge('slow', 'next')
      .edge('next', GRAPH_FINISH)
      .build();

    await expect(graph.run(1, { signal: ac.signal })).rejects.toMatchObject({
      code: 'cancelled',
    });
  });

  test('propagates node errors and fires fail hooks', async () => {
    const events: string[] = [];
    const graph = GraphWorkflow.builder('err')
      .node('boom', () => {
        throw new Error('node failed');
      })
      .edge(GRAPH_START, 'boom')
      .edge('boom', GRAPH_FINISH)
      .build();

    await expect(
      graph.run(null, {
        hooks: {
          onNodeFail: (e) => {
            events.push(`node:${e.nodeId}`);
          },
          onGraphFail: () => {
            events.push('graph');
          },
        },
      }),
    ).rejects.toThrow('node failed');
    expect(events).toEqual(['node:boom', 'graph']);
  });

  test('lifecycle hooks for success path', async () => {
    const events: string[] = [];
    const graph = GraphWorkflow.builder<string, string>('hooks')
      .node('a', (s: string) => s.toUpperCase())
      .edge(GRAPH_START, 'a')
      .edge('a', GRAPH_FINISH)
      .build();

    const result = await graph.run('ok', {
      hooks: {
        onGraphStart: () => {
          events.push('g-start');
        },
        onNodeStart: (e) => {
          events.push(`n-start:${e.nodeId}`);
        },
        onNodeComplete: (e) => {
          events.push(`n-done:${e.nodeId}`);
        },
        onGraphComplete: () => {
          events.push('g-done');
        },
      },
    });
    expect(result.output).toBe('OK');
    expect(events[0]).toBe('g-start');
    expect(events.at(-1)).toBe('g-done');
    expect(events).toContain('n-start:a');
    expect(events).toContain('n-done:a');
  });

  test('nodes can use injected ChatClient via context', async () => {
    const model = new ScriptedChatModel([{ respond: 'hello from model' }]);
    const client = ChatClient.create(model);
    const graph = GraphWorkflow.builder<string, string>('chat-node')
      .node('ask', async (msg: string, ctx) => {
        const clientForNode = ctx.chatClient;
        if (!clientForNode) throw new Error('expected chatClient');
        const content = await clientForNode.prompt().user(msg).call().content();
        return content ?? '';
      })
      .edge(GRAPH_START, 'ask')
      .edge('ask', GRAPH_FINISH)
      .withDefaults({ chatClient: client })
      .build();

    const result = await graph.run('ping');
    expect(result.output).toBe('hello from model');
  });

  test('chatToolLoopGraph runs tool-calling via ChatClient', async () => {
    const weather = functionToolCallback({
      name: 'getWeather',
      description: 'weather',
      inputSchema: {
        type: 'object',
        properties: { city: { type: 'string' } },
        required: ['city'],
      },
      call: ({ city }: { city: string }) => ({ temp: 72, city }),
    });

    const model = new ScriptedChatModel([
      {
        respond: toolCallResponse([toolCall('c1', 'getWeather', { city: 'Yorktown' })]),
      },
      { respond: textResponse('72F in Yorktown') },
    ]);
    const client = ChatClient.create(model);
    const graph = chatToolLoopGraph({
      chatClient: client,
      tools: [weather],
      system: 'Help with weather.',
    });

    const result = await graph.run({ message: 'Weather in Yorktown?' });
    expect(result.output.content).toBe('72F in Yorktown');
  });

  test('simpleAgentGraph returns final text', async () => {
    const model = new ScriptedChatModel([{ respond: 'done' }]);
    const graph = simpleAgentGraph({
      chatClient: ChatClient.create(model),
      system: 'You are concise.',
    });
    const result = await graph.run('hi');
    expect(result.output).toBe('done');
  });

  test('GraphWorkflow.of factory', async () => {
    const graph = GraphWorkflow.of<number, number>('of', (b) => {
      b.node('id', (n: number) => n)
        .edge(GRAPH_START, 'id')
        .edge('id', GRAPH_FINISH);
    });
    expect((await graph.run(7)).output).toBe(7);
  });

  test('async predicates', async () => {
    const graph = GraphWorkflow.builder<number, string>('async-pred')
      .node('x', (n: number) => n)
      .edge(GRAPH_START, 'x')
      .edge('x', GRAPH_FINISH, {
        when: async (n: number) => {
          await Bun.sleep(1);
          return n > 0;
        },
        transform: async () => 'yes',
      })
      .edge('x', GRAPH_FINISH, {
        when: async () => true,
        transform: async () => 'no',
      })
      .build();
    expect((await graph.run(1)).output).toBe('yes');
  });

  test('graphName getter, validate(), and link() convenience edge', async () => {
    const graph = GraphWorkflow.builder<number, number>('named')
      .node('x', (n: number) => n)
      .edge(GRAPH_START, 'x')
      .link('x', GRAPH_FINISH, 'to-finish')
      .build();

    expect(graph.graphName).toBe('named');
    expect(() => graph.validate()).not.toThrow();
    const result = await graph.run(5);
    expect(result.output).toBe(5);
    expect(result.steps.at(-1)?.edgeName).toBe('to-finish');
  });

  test('graphWorkflow() module factory produces a working builder', async () => {
    const graph = graphWorkflow<number, number>('factory-made')
      .node('id', (n: number) => n)
      .edge(GRAPH_START, 'id')
      .edge('id', GRAPH_FINISH)
      .build();
    expect((await graph.run(9)).output).toBe(9);
  });

  test('withDefaults hooks merge with per-run hooks (both fire in order)', async () => {
    const events: string[] = [];
    const graph = GraphWorkflow.builder<string, string>('merged-hooks')
      .node('a', (s: string) => s.toUpperCase())
      .edge(GRAPH_START, 'a')
      .edge('a', GRAPH_FINISH)
      .withDefaults({
        hooks: {
          onGraphStart: () => {
            events.push('default:g-start');
          },
          onNodeComplete: (e) => {
            events.push(`default:n-done:${e.nodeId}`);
          },
        },
      })
      .build();

    const result = await graph.run('ok', {
      hooks: {
        onGraphStart: () => {
          events.push('run:g-start');
        },
        onNodeComplete: (e) => {
          events.push(`run:n-done:${e.nodeId}`);
        },
      },
    });

    expect(result.output).toBe('OK');
    expect(events[0]).toBe('default:g-start');
    expect(events[1]).toBe('run:g-start');
    expect(events).toContain('default:n-done:a');
    expect(events).toContain('run:n-done:a');
  });
});

describe('GraphWorkflow errors', () => {
  test('isAiError for validation failures', () => {
    try {
      GraphWorkflow.builder('x').build();
      expect.unreachable();
    } catch (e) {
      expect(isAiError(e)).toBe(true);
    }
  });
});
