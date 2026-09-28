import { describe, expect, it } from 'bun:test';
import { ChatAgent } from '../../src/agent/chat-agent.ts';
import {
  A2ADirectory,
  type A2AFetchLike,
  Agent,
  AgentCardHelper,
  AiTokens,
  configureAi,
  createA2AHttpHandler,
  createChatAgentA2AExecutor,
  ScriptedChatModel,
} from '../../src/index.ts';

interface WorkTicket {
  id: string;
  skill: string;
  spec: string;
}

interface WorkResult {
  ticketId: string;
  skill: string;
  status: string;
  summary: string;
}

describe('A2A Multi-Origin Protocol Flow (Forge, Aria, Ravi)', () => {
  it('Forge manager dispatches Work to Aria and Ravi across independent HTTP origins without importing internal classes or MCP tools', async () => {
    // 1. Aria Service Origin (http://aria.local)
    const ariaModel = new ScriptedChatModel([
      { respond: 'LGTM: All code conventions and safety checks passed.' },
    ]);
    const ariaAgent = ChatAgent.create({
      chatModel: ariaModel,
      system: 'Aria Senior Reviewer Agent',
    });
    const ariaCard = AgentCardHelper.create({
      name: 'AriaReviewer',
      description: 'Performs deep automated code review',
      a2a: { url: 'http://aria.local/rpc' },
      skills: [{ id: 'dev.review', description: 'Review code changes against repo standards' }],
    });
    const ariaHandler = createA2AHttpHandler({
      card: ariaCard,
      executor: createChatAgentA2AExecutor(ariaAgent, {
        createDefaultArtifact: true,
      }),
    });

    // 2. Ravi Service Origin (http://ravi.local)
    const raviModel = new ScriptedChatModel([
      { respond: 'PASS: 42 unit tests ran in 12ms. 0 failures.' },
    ]);
    const raviAgent = ChatAgent.create({
      chatModel: raviModel,
      system: 'Ravi Quality Assurance Agent',
    });
    const raviCard = AgentCardHelper.create({
      name: 'RaviQA',
      description: 'Executes automated test suites',
      a2a: { url: 'http://ravi.local/rpc' },
      skills: [{ id: 'dev.test', description: 'Execute integration and unit test suites' }],
    });
    const raviHandler = createA2AHttpHandler({
      card: raviCard,
      executor: createChatAgentA2AExecutor(raviAgent, {
        createDefaultArtifact: true,
      }),
    });

    // 3. Network Transport Router (Loopback HTTP Router simulating network requests)
    const networkFetch: A2AFetchLike = async (input, init) => {
      const url = typeof input === 'string' ? input : input.toString();
      const req = new Request(url, init);
      if (url.includes('aria.local')) {
        return ariaHandler(req);
      }
      if (url.includes('ravi.local')) {
        return raviHandler(req);
      }
      return new Response(JSON.stringify({ error: 'Origin not found' }), { status: 404 });
    };

    // 4. Forge Manager Origin (http://forge.local)
    // Forge resolves A2ADirectory and dispatches tasks based strictly on skill ID
    const forgeDirectory = A2ADirectory.create({
      origins: ['http://aria.local', 'http://ravi.local'],
      fetch: networkFetch,
    });

    class ForgeManagerService {
      constructor(private readonly directory: A2ADirectory) {}

      async dispatchWork(ticket: WorkTicket): Promise<WorkResult> {
        // Discovers peer by skill without importing their agent class or MCP tool references
        const client = await this.directory.find({ skill: ticket.skill });
        const task = await client.sendAndWait({
          skill: ticket.skill,
          message: ticket.spec,
          metadata: { workId: ticket.id },
        });

        const lastMsg = task.history?.[task.history.length - 1];
        const textPart = lastMsg?.parts[0] as { text: string } | undefined;

        return {
          ticketId: ticket.id,
          skill: ticket.skill,
          status: task.status.state,
          summary: textPart?.text ?? '',
        };
      }
    }

    const forgeManager = new ForgeManagerService(forgeDirectory);

    // 5. Execute Dispatch Flow
    const reviewResult = await forgeManager.dispatchWork({
      id: 'work-101',
      skill: 'dev.review',
      spec: 'git diff main...feature',
    });

    expect(reviewResult.ticketId).toBe('work-101');
    expect(reviewResult.skill).toBe('dev.review');
    expect(reviewResult.status).toBe('completed');
    expect(reviewResult.summary).toContain('LGTM');

    const testResult = await forgeManager.dispatchWork({
      id: 'work-102',
      skill: 'dev.test',
      spec: 'run all tests',
    });

    expect(testResult.ticketId).toBe('work-102');
    expect(testResult.skill).toBe('dev.test');
    expect(testResult.status).toBe('completed');
    expect(testResult.summary).toContain('PASS');
  });
});
