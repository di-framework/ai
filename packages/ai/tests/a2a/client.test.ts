import { describe, expect, it } from 'bun:test';
import {
  type A2AAgentExecutor,
  A2AClient,
  A2ATaskStore,
  AgentCardHelper,
  createA2AHttpHandler,
  createTextMessage,
} from '../../src/a2a/index.ts';

describe('A2AClient', () => {
  const card = AgentCardHelper.create({
    name: 'EchoAgent',
    description: 'Echoes input back',
    version: '1.0.0',
    a2a: { url: 'http://localhost/a2a/rpc' },
    skills: [{ id: 'echo.skill', description: 'Echo back' }],
  });

  const executor: A2AAgentExecutor = {
    async execute(_task, message, _ctx) {
      const text = (message.parts[0] as { text: string }).text;
      return {
        status: { state: 'completed' },
        messages: [createTextMessage(`Echo: ${text}`, 'agent')],
      };
    },
  };

  const handler = createA2AHttpHandler({ card, executor });

  const customFetch = async (input: string | URL, init?: RequestInit): Promise<Response> => {
    const urlStr = typeof input === 'string' ? input : input.toString();
    return handler(new Request(urlStr, init));
  };

  it('fetches agent card and discovers JSON-RPC interface', async () => {
    const client = A2AClient.of('http://localhost', {
      fetch: customFetch,
      headers: new Headers({ 'X-Custom': 'val' }),
    });

    const discoveredCard = await client.getCard();
    expect(discoveredCard.name).toBe('EchoAgent');
    expect(discoveredCard.skills.length).toBe(1);

    const url = await client.discoverEndpoint();
    expect(url).toBe('http://localhost/a2a/rpc');
  });

  it('supports array headers and function headers', async () => {
    const client = A2AClient.create({
      baseUrl: 'http://localhost',
      fetch: customFetch,
      headers: async () => [['X-Key', 'Value']],
    });

    const res = await client.send({
      message: 'Array headers test',
    });
    expect(res.task?.status.state).toBe('completed');
  });

  it('sends a message and receives the completed task', async () => {
    const client = A2AClient.create({
      baseUrl: 'http://localhost',
      fetch: customFetch,
    });

    const res = await client.send({
      skill: 'echo.skill',
      message: 'Ping!',
      metadata: { traceId: 'tr-123' },
    });

    expect(res.task).toBeDefined();
    expect(res.task?.status.state).toBe('completed');
    const msg = res.task?.history?.[1]?.parts[0] as { text: string } | undefined;
    expect(msg?.text).toBe('Echo: Ping!');
  });

  it('sendAndWait completes a task with polling if needed', async () => {
    const taskStore = A2ATaskStore.create();
    const workingExecutor: A2AAgentExecutor = {
      async execute(task, _msg, _ctx) {
        // Complete the task in the background after 20ms
        setTimeout(() => {
          taskStore.updateStatus(
            task.id,
            'completed',
            createTextMessage('Background done', 'agent'),
          );
        }, 20);
        return { status: { state: 'working' } };
      },
    };

    const workingHandler = createA2AHttpHandler({ card, executor: workingExecutor, taskStore });

    const client = A2AClient.create({
      baseUrl: 'http://localhost',
      fetch: async (input: string | URL, init?: RequestInit) => {
        return workingHandler(
          new Request(typeof input === 'string' ? input : input.toString(), init),
        );
      },
    });

    const task = await client.sendAndWait({
      message: 'Run background',
      pollIntervalMs: 10,
    });

    expect(task.status.state).toBe('completed');
  });

  it('fails with clear AiError on protocol version mismatch', async () => {
    const badCard = {
      ...card,
      supported_interfaces: [
        {
          transport: 'HTTP' as const,
          url: 'http://localhost/a2a/rpc',
          protocol_version: '2.0',
          protocol_binding: 'JSONRPC' as const,
        },
      ],
    };

    const badHandler = createA2AHttpHandler({ card: badCard, executor });

    const client = A2AClient.create({
      baseUrl: 'http://localhost',
      fetch: async (input: string | URL, init?: RequestInit) => {
        return badHandler(new Request(typeof input === 'string' ? input : input.toString(), init));
      },
    });

    expect(client.send({ message: 'Hello' })).rejects.toThrow();
  });

  it('lists and cancels tasks', async () => {
    const client = A2AClient.create({
      baseUrl: 'http://localhost',
      fetch: customFetch,
    });

    const sendRes = await client.send({ message: 'Task to cancel' });
    expect(sendRes.task).toBeDefined();

    const fetchedTask = await client.getTask(sendRes.task?.id ?? '', { history: true });
    expect(fetchedTask.id).toBe(sendRes.task?.id ?? '');

    const list = await client.listTasks({ limit: 10 });
    expect(list.tasks.length).toBeGreaterThan(0);

    const canceled = await client.cancel(sendRes.task?.id ?? '', 'User cancel');
    expect(canceled.id).toBe(sendRes.task?.id ?? '');
  });

  it('normalizes task with artifacts, name, description, mimeType, uri, and parts', async () => {
    const artExecutor: A2AAgentExecutor = {
      async execute(_task, _msg, _ctx) {
        return {
          status: { state: 'completed' },
          artifacts: [
            {
              artifactId: 'art-1',
              name: 'Artifact 1',
              description: 'Sample description',
              mimeType: 'text/plain',
              uri: 'file:///tmp/art1.txt',
              parts: [{ kind: 'text', text: 'Hello Artifact' }],
            },
          ],
        };
      },
    };

    const artHandler = createA2AHttpHandler({ card, executor: artExecutor });
    const client = A2AClient.create({
      baseUrl: 'http://localhost',
      fetch: async (input: string | URL, init?: RequestInit) => {
        return artHandler(new Request(typeof input === 'string' ? input : input.toString(), init));
      },
    });

    const res = await client.send({ message: 'Generate artifact' });
    expect(res.task?.artifacts?.[0]?.artifactId).toBe('art-1');
    expect(res.task?.artifacts?.[0]?.name).toBe('Artifact 1');
    expect(res.task?.artifacts?.[0]?.description).toBe('Sample description');
    expect(res.task?.artifacts?.[0]?.mimeType).toBe('text/plain');
    expect(res.task?.artifacts?.[0]?.uri).toBe('file:///tmp/art1.txt');
    expect(res.task?.artifacts?.[0]?.parts?.length).toBe(1);
  });
});
