import type { ChatClient } from '../chat/client/default-chat-client.ts';
import {
  callChatContent,
  callChatEntity,
  mapPool,
  type WorkflowCallOptions,
} from './workflow-utils.ts';

export interface WorkerTask {
  readonly type: string;
  readonly description: string;
}

export interface OrchestratorPlan {
  readonly analysis: string;
  readonly tasks: readonly WorkerTask[];
}

export interface WorkerResponse {
  readonly type: string;
  readonly description: string;
  readonly result: string;
}

export interface OrchestratorWorkersResult {
  readonly analysis: string;
  readonly workerResponses: readonly WorkerResponse[];
  /** Optional synthesized answer when {@link OrchestratorWorkersWorkflowOptions.synthesize} is true. */
  readonly finalResponse?: string;
}

export interface OrchestratorWorkersWorkflowOptions extends WorkflowCallOptions {
  readonly concurrency?: number;
  /**
   * When true (default), run a final synthesis step over worker outputs.
   */
  readonly synthesize?: boolean;
  readonly orchestratorSystem?: string;
  readonly workerSystem?: string;
  readonly synthesizerSystem?: string;
}

const PLAN_SCHEMA = {
  type: 'object',
  properties: {
    analysis: { type: 'string' },
    tasks: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          type: { type: 'string' },
          description: { type: 'string' },
        },
        required: ['type', 'description'],
      },
    },
  },
  required: ['analysis', 'tasks'],
} as const;

const DEFAULT_ORCHESTRATOR = [
  'You are an orchestrator that breaks a complex task into independent worker subtasks.',
  'Return JSON only with shape:',
  '{"analysis": string, "tasks": [{"type": string, "description": string}, ...]}',
  'Create 2-5 focused tasks. Each task description must be self-contained for a worker.',
].join('\n');

const DEFAULT_WORKER =
  'You are a focused worker. Complete only the assigned subtask. Be concrete and concise.';

const DEFAULT_SYNTHESIZER = [
  'You synthesize worker outputs into a single coherent final answer for the original task.',
  'Use the analysis and each worker result. Do not invent work that was not done.',
].join('\n');

/**
 * Orchestrator plans subtasks; workers execute them (optionally in parallel);
 * optional synthesizer produces a final answer.
 * Spring AI / Anthropic: Orchestrator-Workers.
 */
export class OrchestratorWorkersWorkflow {
  private readonly chatClient: ChatClient;

  constructor(chatClient: ChatClient) {
    this.chatClient = chatClient;
  }

  /** Factory alias (same as {@link orchestratorWorkersWorkflow}). */
  static of(chatClient: ChatClient): OrchestratorWorkersWorkflow {
    return new OrchestratorWorkersWorkflow(chatClient);
  }

  async process(
    taskDescription: string,
    options?: OrchestratorWorkersWorkflowOptions,
  ): Promise<OrchestratorWorkersResult> {
    const plan = await callChatEntity<OrchestratorPlan>(this.chatClient, {
      system: options?.orchestratorSystem ?? DEFAULT_ORCHESTRATOR,
      user: taskDescription,
      schema: PLAN_SCHEMA as unknown as Record<string, unknown>,
      signal: options?.signal,
      options: options?.options,
      map: (v) => {
        const obj = v as OrchestratorPlan;
        return {
          analysis: String(obj.analysis ?? ''),
          tasks: Array.isArray(obj.tasks)
            ? obj.tasks.map((t) => ({
                type: String(t.type ?? 'task'),
                description: String(t.description ?? ''),
              }))
            : [],
        };
      },
    });

    const workerSystem = options?.workerSystem ?? DEFAULT_WORKER;
    const concurrency = options?.concurrency ?? (plan.tasks.length > 0 ? plan.tasks.length : 1);

    const workerResponses = await mapPool(
      plan.tasks,
      concurrency,
      async (task) => {
        const result = await callChatContent(this.chatClient, {
          system: workerSystem,
          user: [
            `Task type: ${task.type}`,
            `Task: ${task.description}`,
            `Original request: ${taskDescription}`,
            `Orchestrator analysis: ${plan.analysis}`,
          ].join('\n'),
          signal: options?.signal,
          options: options?.options,
        });
        return {
          type: task.type,
          description: task.description,
          result,
        } satisfies WorkerResponse;
      },
      options?.signal,
    );

    const synthesize = options?.synthesize ?? true;
    let finalResponse: string | undefined;
    if (synthesize && workerResponses.length > 0) {
      const synthesisUser = [
        `Original task: ${taskDescription}`,
        `Analysis: ${plan.analysis}`,
        'Worker results:',
        ...workerResponses.map((w, i) => `${i + 1}. [${w.type}] ${w.description}\n${w.result}`),
      ].join('\n\n');
      finalResponse = await callChatContent(this.chatClient, {
        system: options?.synthesizerSystem ?? DEFAULT_SYNTHESIZER,
        user: synthesisUser,
        signal: options?.signal,
        options: options?.options,
      });
    }

    return {
      analysis: plan.analysis,
      workerResponses,
      finalResponse,
    };
  }
}

export function orchestratorWorkersWorkflow(chatClient: ChatClient): OrchestratorWorkersWorkflow {
  return OrchestratorWorkersWorkflow.of(chatClient);
}
