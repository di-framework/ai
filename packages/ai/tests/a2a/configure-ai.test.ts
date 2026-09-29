import { describe, expect, it } from 'bun:test';
import {
  A2AClient,
  A2ADirectory,
  A2ATaskStore,
  Agent,
  AiTokens,
  configureAi,
  EnableAi,
  enableAi,
  ScriptedChatModel,
} from '../../src/index.ts';

describe('configureAi and @EnableAi with A2A', () => {
  @Agent({
    name: 'ScannedAria',
    description: 'Reviews pull requests',
    a2a: { url: 'http://scanned-aria.local/a2a' },
    skills: [{ id: 'dev.review', description: 'Review diffs' }],
  })
  class ScannedAriaAgent {}

  @EnableAi({
    a2a: {
      origins: ['http://scanned-aria.local'],
    },
  })
  class SampleApp {}

  it('registers A2ADirectory and A2ATaskStore when a2a: true', async () => {
    const model = new ScriptedChatModel([{ respond: 'Reviewed!' }]);
    const result = await configureAi({
      chatModel: model,
      a2a: {
        origins: ['http://origin1.local', 'http://origin2.local'],
      },
    });

    const directory = result.container.resolve<A2ADirectory>(AiTokens.A2A_DIRECTORY);
    expect(directory).toBeDefined();
    expect(directory.getOrigins()).toEqual(['http://origin1.local', 'http://origin2.local']);

    const directoryByClass = result.container.resolve<A2ADirectory>(A2ADirectory);
    expect(directoryByClass).toBeDefined();

    const taskStore = result.container.resolve<A2ATaskStore>(AiTokens.A2A_TASK_STORE);
    expect(taskStore).toBeDefined();
    expect(taskStore instanceof A2ATaskStore).toBe(true);
  });

  it('serves @Agent({ a2a: { url } }) over HTTP handler when a2a is enabled', async () => {
    const model = new ScriptedChatModel([{ respond: 'Code looks clean and ready to merge.' }]);

    const result = await configureAi({
      chatModel: model,
      a2a: true,
      scanAnnotations: true,
    });

    const handler = result.container.resolve<(req: Request) => Promise<Response>>(
      AiTokens.A2A_HTTP_HANDLER,
    );
    expect(handler).toBeDefined();

    // Loopback test against the automatically registered HTTP handler
    const client = A2AClient.create({
      baseUrl: 'http://scanned-aria.local',
      fetch: async (input: string | URL, init?: RequestInit) => {
        return handler(new Request(typeof input === 'string' ? input : input.toString(), init));
      },
    });

    const card = await client.getCard();
    expect(card.name).toBe('ScannedAria');
    expect(card.skills[0]?.id).toBe('dev.review');

    const sendRes = await client.send({
      skill: 'dev.review',
      message: 'Diff check',
    });
    expect(sendRes.task?.status.state).toBe('completed');
  });

  it('enableAi merges options and sets up A2ADirectory', async () => {
    const model = new ScriptedChatModel([{ respond: 'OK' }]);
    const result = enableAi(SampleApp, { chatModel: model });

    const directory = result.container.resolve<A2ADirectory>(A2ADirectory);
    expect(directory.getOrigins()).toContain('http://scanned-aria.local');
  });
});
