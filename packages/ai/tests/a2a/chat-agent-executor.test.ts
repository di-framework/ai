import { describe, expect, it } from 'bun:test';
import {
  A2AClient,
  AgentCardHelper,
  ChatAgentA2AExecutor,
  createA2AHttpHandler,
  createChatAgentA2AExecutor,
} from '../../src/a2a/index.ts';
import { ChatAgent } from '../../src/agent/chat-agent.ts';
import { ScriptedChatModel } from '../../src/testing/fake-chat-model.ts';
import { functionToolCallback } from '../../src/tool/function-tool-callback.ts';

describe('ChatAgent as A2A Executor', () => {
  it('completes an A2A Task and returns an Artifact via ChatAgent with tools', async () => {
    // 1. Scripted ChatModel that uses internal tool
    const scriptedModel = new ScriptedChatModel([
      {
        respond: 'Review analysis: All checks passed with 0 vulnerabilities.',
      },
    ]);

    // 2. Internal tool (MCP / @Tool style)
    const localTool = functionToolCallback({
      name: 'internalLinter',
      description: 'Lints code internally',
      call: async () => 'Clean code',
    });

    // 3. Process-local ChatAgent
    const chatAgent = ChatAgent.create({
      chatModel: scriptedModel,
      system: 'You are a private static analysis agent with super secret prompt guidelines.',
      tools: [localTool],
    });

    // Test ChatAgentA2AExecutor.of
    const directExecutor = ChatAgentA2AExecutor.of(chatAgent);
    expect(directExecutor).toBeDefined();

    // 4. Adapt to A2A executor
    const executor = createChatAgentA2AExecutor(chatAgent, {
      artifactExtractor: (content) => [
        {
          artifactId: 'art-review-report',
          name: 'review.md',
          mimeType: 'text/markdown',
          parts: [{ kind: 'text', text: content }],
        },
      ],
    });

    // 5. Card and HTTP handler
    const card = AgentCardHelper.create({
      name: 'ReviewAgent',
      description: 'Performs code reviews',
      a2a: { url: 'http://review-agent.local/rpc' },
      skills: [{ id: 'dev.review', description: 'Review pull request diffs' }],
    });

    const handler = createA2AHttpHandler({ card, executor });

    const client = A2AClient.create({
      baseUrl: 'http://review-agent.local',
      fetch: async (input: string | URL, init?: RequestInit) => {
        const urlStr = typeof input === 'string' ? input : input.toString();
        return handler(new Request(urlStr, init));
      },
    });

    // 6. Client dispatches message
    const result = await client.sendAndWait({
      skill: 'dev.review',
      message: 'PR diff: + const a = 1;',
    });

    expect(result.status.state).toBe('completed');
    expect(result.artifacts?.length).toBe(1);
    expect(result.artifacts?.[0]?.artifactId).toBe('art-review-report');
    expect((result.artifacts?.[0]?.parts?.[0] as { text: string })?.text).toContain(
      'All checks passed',
    );

    // 7. Verify Opacity: no tools or system prompt leaked on wire
    const cardJson = JSON.stringify(await client.getCard());
    expect(cardJson).not.toContain('internalLinter');
    expect(cardJson).not.toContain('super secret prompt guidelines');

    const taskJson = JSON.stringify(result);
    expect(taskJson).not.toContain('internalLinter');
    expect(taskJson).not.toContain('super secret prompt guidelines');
  });
});
