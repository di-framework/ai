import { describe, expect, it } from 'bun:test';
import {
  type A2AAgentExecutor,
  A2ADirectory,
  AgentCardHelper,
  createA2AHttpHandler,
  createTextMessage,
} from '../../src/a2a/index.ts';
import { isAiError } from '../../src/model/errors.ts';

describe('A2ADirectory', () => {
  const ariaCard = AgentCardHelper.create({
    name: 'AriaReviewer',
    description: 'Reviews code',
    a2a: { url: 'http://aria.local/rpc' },
    skills: [{ id: 'dev.review', description: 'Review diffs' }],
  });

  const raviCard = AgentCardHelper.create({
    name: 'RaviTester',
    description: 'Runs tests',
    a2a: { url: 'http://ravi.local/rpc' },
    skills: [{ id: 'dev.test', description: 'Run test suite' }],
  });

  const ariaExecutor: A2AAgentExecutor = {
    async execute(_task, _message, _ctx) {
      return {
        status: { state: 'completed' },
        messages: [createTextMessage('Aria review complete', 'agent')],
      };
    },
  };

  const raviExecutor: A2AAgentExecutor = {
    async execute(_task, _message, _ctx) {
      return {
        status: { state: 'completed' },
        messages: [createTextMessage('Ravi tests passed', 'agent')],
      };
    },
  };

  const ariaHandler = createA2AHttpHandler({ card: ariaCard, executor: ariaExecutor });
  const raviHandler = createA2AHttpHandler({ card: raviCard, executor: raviExecutor });

  const routerFetch = async (input: string | URL, init?: RequestInit): Promise<Response> => {
    const urlStr = typeof input === 'string' ? input : input.toString();
    const req = new Request(urlStr, init);
    if (urlStr.includes('aria.local')) {
      return ariaHandler(req);
    }
    if (urlStr.includes('ravi.local')) {
      return raviHandler(req);
    }
    return new Response(JSON.stringify({ error: 'Not Found' }), { status: 404 });
  };

  it('finds agent client by skill from registered origins', async () => {
    const directory = A2ADirectory.of('http://aria.local', 'http://ravi.local');
    const directoryWithFetch = A2ADirectory.create({
      origins: ['http://aria.local', 'http://ravi.local'],
      fetch: routerFetch,
    });

    const reviewer = await directoryWithFetch.find({ skill: 'dev.review' });
    expect(reviewer).toBeDefined();

    const reviewRes = await reviewer.send({
      skill: 'dev.review',
      message: 'Diff for review',
    });
    expect(reviewRes.task?.status.state).toBe('completed');
    const ariaMsgPart = reviewRes.task?.history?.[1]?.parts[0] as { text: string } | undefined;
    expect(ariaMsgPart?.text).toBe('Aria review complete');

    const tester = await directoryWithFetch.find({ skill: 'dev.test' });
    expect(tester).toBeDefined();

    const testRes = await tester.send({
      skill: 'dev.test',
      message: 'Test run',
    });
    expect(testRes.task?.status.state).toBe('completed');
    const raviMsgPart = testRes.task?.history?.[1]?.parts[0] as { text: string } | undefined;
    expect(raviMsgPart?.text).toBe('Ravi tests passed');
  });

  it('findAll returns all matching clients or all clients', async () => {
    const directory = A2ADirectory.create({
      origins: ['http://aria.local', 'http://ravi.local'],
      fetch: routerFetch,
    });

    const all = await directory.findAll();
    expect(all.length).toBe(2);

    const matching = await directory.findAll({ skill: 'dev.review' });
    expect(matching.length).toBe(1);

    directory.clearCache();
  });

  it('fails with AiError when requested skill is missing', async () => {
    const directory = A2ADirectory.create({
      origins: ['http://aria.local', 'http://ravi.local'],
      fetch: routerFetch,
    });

    expect(directory.find({ skill: 'dev.deploy' })).rejects.toThrow();
    try {
      await directory.find({ skill: 'dev.deploy' });
    } catch (err) {
      expect(isAiError(err)).toBe(true);
      if (isAiError(err)) {
        expect(err.code).toBe('invalid-request');
      }
    }
  });

  it('allows dynamic registration and unregistration of origins', async () => {
    const directory = A2ADirectory.create({ fetch: routerFetch });
    expect(directory.getOrigins().length).toBe(0);

    directory.add('http://aria.local');
    expect(directory.getOrigins()).toEqual(['http://aria.local']);

    const reviewer = await directory.find({ skill: 'dev.review' });
    expect(reviewer).toBeDefined();

    directory.remove('http://aria.local');
    expect(directory.getOrigins().length).toBe(0);
    expect(directory.find({ skill: 'dev.review' })).rejects.toThrow();
  });
});
