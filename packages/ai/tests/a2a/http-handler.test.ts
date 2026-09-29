import { describe, expect, it } from 'bun:test';
import {
  type A2AAgentExecutor,
  AgentCardHelper,
  createA2AHttpHandler,
  createTextMessage,
} from '../../src/a2a/index.ts';

describe('createA2AHttpHandler', () => {
  const card = AgentCardHelper.create({
    name: 'TestAgent',
    description: 'A test agent',
    version: '1.0.0',
    a2a: { url: 'http://localhost/a2a/rpc' },
    skills: [{ id: 'test.skill', description: 'Test skill' }],
  });

  const executor: A2AAgentExecutor = {
    async execute(_task, message, _ctx) {
      const textPart = message.parts[0] as { text: string };
      return {
        status: { state: 'completed' },
        messages: [createTextMessage(`Processed: ${textPart.text}`, 'agent')],
      };
    },
  };

  it('serves GET /.well-known/agent-card.json', async () => {
    const handler = createA2AHttpHandler({ card, executor });

    const req = new Request('http://localhost/.well-known/agent-card.json', { method: 'GET' });
    const res = await handler(req);

    expect(res.status).toBe(200);
    expect(res.headers.get('Content-Type')).toBe('application/json');

    const body = (await res.json()) as { name: string; skills: unknown[] };
    expect(body.name).toBe('TestAgent');
    expect(body.skills.length).toBe(1);
  });

  it('handles JSON-RPC SendMessage over POST', async () => {
    const handler = createA2AHttpHandler({ card, executor });

    const req = new Request('http://localhost/a2a/rpc', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: 'req-1',
        method: 'SendMessage',
        params: {
          message: createTextMessage('Hello World', 'user'),
        },
      }),
    });

    const res = await handler(req);
    expect(res.status).toBe(200);

    const body = (await res.json()) as { result: { task: { status: { state: string } } } };
    expect(body.result.task.status.state).toBe('completed');
  });

  it('handles authHandler returning false, true, or response', async () => {
    const rejectingHandler = createA2AHttpHandler({
      card,
      executor,
      authHandler: async () => false,
    });

    const req = new Request('http://localhost/a2a/rpc', {
      method: 'POST',
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: 1,
        method: 'SendMessage',
        params: { message: createTextMessage('Hi', 'user') },
      }),
    });
    const res = await rejectingHandler(req);
    expect(res.status).toBe(401);

    const customResHandler = createA2AHttpHandler({
      card,
      executor,
      authHandler: async () => new Response('Custom Block', { status: 403 }),
    });
    const res2 = await customResHandler(req);
    expect(res2.status).toBe(403);

    const allowHandler = createA2AHttpHandler({
      card,
      executor,
      authHandler: async () => true,
    });
    const res3 = await allowHandler(req);
    expect(res3.status).toBe(200);
  });

  it('returns 404 for unknown GET routes and 405 for unsupported HTTP methods', async () => {
    const handler = createA2AHttpHandler({ card, executor });

    const getReq = new Request('http://localhost/unknown-path', { method: 'GET' });
    const getRes = await handler(getReq);
    expect(getRes.status).toBe(404);

    const putReq = new Request('http://localhost/a2a/rpc', { method: 'PUT' });
    const putRes = await handler(putReq);
    expect(putRes.status).toBe(405);
  });

  it('handles trailing slashes on basePath and request URL', async () => {
    const handler = createA2AHttpHandler({ card, executor, basePath: '/api///' });
    const res = await handler(new Request('http://localhost/api/.well-known/agent-card.json///'));
    expect(res.status).toBe(200);
  });
});
