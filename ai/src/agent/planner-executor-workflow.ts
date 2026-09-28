import type { ChatClient, ToolSource } from '../chat/client/default-chat-client.ts';
import type { ChatOptions } from '../chat/prompt/chat-options.ts';
import { AiError } from '../model/errors.ts';
import { resolveToolCallbacks } from '../tool/tool-callback-provider.ts';
import {
  callChatContent,
  callChatEntity,
  throwIfAborted,
  type WorkflowCallOptions,
} from './workflow-utils.ts';

export interface PlannerStep {
  readonly id: string;
  readonly description: string;
  /** When set, the executor prefers invoking this named tool via the tool-aware client. */
  readonly toolName?: string;
  readonly status: 'pending' | 'done' | 'failed' | 'skipped';
  readonly result?: string;
}

export interface PlannerPlan {
  readonly goal: string;
  readonly reasoning?: string;
  readonly steps: readonly PlannerStep[];
  /** When true, planning is complete and {@link finalAnswer} (if any) should be returned. */
  readonly done: boolean;
  readonly finalAnswer?: string;
}

export interface PlannerRound {
  readonly index: number;
  readonly phase: 'plan' | 'act' | 'replan';
  readonly plan: PlannerPlan;
  readonly action?: {
    readonly stepId: string;
    readonly observation: string;
  };
}

export interface PlannerExecutorResult {
  readonly answer: string;
  readonly plan: PlannerPlan;
  readonly rounds: readonly PlannerRound[];
  readonly stepCount: number;
}

export interface PlannerExecutorOptions extends WorkflowCallOptions {
  /**
   * Max plan/act cycles (each act counts as one). Default 8.
   */
  readonly maxSteps?: number;
  /**
   * Tools available during act phase (ChatClient tool-calling).
   */
  readonly tools?: readonly ToolSource[];
  readonly plannerSystem?: string;
  readonly actorSystem?: string;
  /**
   * Optional seed plan; when omitted the first call is a full plan.
   */
  readonly initialPlan?: PlannerPlan;
}

const PLAN_SCHEMA = {
  type: 'object',
  properties: {
    goal: { type: 'string' },
    reasoning: { type: 'string' },
    done: { type: 'boolean' },
    finalAnswer: { type: 'string' },
    steps: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          description: { type: 'string' },
          toolName: { type: 'string' },
          status: {
            type: 'string',
            enum: ['pending', 'done', 'failed', 'skipped'],
          },
          result: { type: 'string' },
        },
        required: ['id', 'description', 'status'],
      },
    },
  },
  required: ['goal', 'done', 'steps'],
} as const;

const DEFAULT_PLANNER = [
  'You are a planner–executor controller.',
  'Given a user goal and optional prior plan/observations, produce an updated plan as JSON only.',
  'Schema:',
  '{"goal": string, "reasoning"?: string, "done": boolean, "finalAnswer"?: string,',
  ' "steps": [{"id": string, "description": string, "toolName"?: string,',
  '   "status": "pending"|"done"|"failed"|"skipped", "result"?: string}]}',
  'Rules:',
  '- Prefer 1–5 concrete pending steps; keep already done steps with their results.',
  '- Set done=true and finalAnswer when the goal is satisfied from observations.',
  '- Do not invent tool results; only mark done when observations support it.',
  '- Use toolName only for tools known to be available.',
].join('\n');

const DEFAULT_ACTOR = [
  'You execute a single planned step toward the user goal.',
  'Use tools when helpful. Return a concise observation of what you did and learned.',
  'Do not claim the overall goal is finished unless this step completes it.',
].join('\n');

const DEFAULT_MAX_STEPS = 8;

function normalizePlan(raw: unknown, fallbackGoal: string): PlannerPlan {
  const obj = (raw ?? {}) as Partial<PlannerPlan>;
  const steps = Array.isArray(obj.steps)
    ? obj.steps.map((s, i) => {
        const step = s as Partial<PlannerStep>;
        const status = step.status;
        const okStatus =
          status === 'pending' || status === 'done' || status === 'failed' || status === 'skipped'
            ? status
            : 'pending';
        return {
          id: String(step.id ?? `s${i + 1}`),
          description: String(step.description ?? ''),
          toolName: step.toolName ? String(step.toolName) : undefined,
          status: okStatus,
          result: step.result !== undefined ? String(step.result) : undefined,
        } satisfies PlannerStep;
      })
    : [];
  return {
    goal: String(obj.goal ?? fallbackGoal),
    reasoning: obj.reasoning !== undefined ? String(obj.reasoning) : undefined,
    steps,
    done: Boolean(obj.done),
    finalAnswer: obj.finalAnswer !== undefined ? String(obj.finalAnswer) : undefined,
  };
}

/** Stable fingerprint for cycle detection (ignores reasoning prose). */
export function planFingerprint(plan: PlannerPlan): string {
  const body = {
    done: plan.done,
    finalAnswer: plan.finalAnswer ?? '',
    steps: plan.steps.map((s) => ({
      id: s.id,
      description: s.description,
      toolName: s.toolName ?? '',
      status: s.status,
      result: s.result ?? '',
    })),
  };
  return JSON.stringify(body);
}

/**
 * Plan → act → replan loop on {@link ChatClient} (+ optional tools).
 *
 * Distinct from {@link OrchestratorWorkersWorkflow} (one-shot fan-out) and from
 * {@link GraphWorkflow} (explicit topology). Use this when the model should
 * iteratively refine a todo list until the goal is done.
 *
 * @example
 * ```ts
 * const pe = new PlannerExecutorWorkflow(chatClient);
 * const { answer } = await pe.run('Book a table for 2', {
 *   tools: [reservationTool],
 *   maxSteps: 6,
 *   signal,
 * });
 * ```
 */
export class PlannerExecutorWorkflow {
  private readonly chatClient: ChatClient;

  constructor(chatClient: ChatClient) {
    this.chatClient = chatClient;
  }

  static of(chatClient: ChatClient): PlannerExecutorWorkflow {
    return new PlannerExecutorWorkflow(chatClient);
  }

  async run(goal: string, options?: PlannerExecutorOptions): Promise<PlannerExecutorResult> {
    throwIfAborted(options?.signal);
    const maxSteps = options?.maxSteps ?? DEFAULT_MAX_STEPS;
    if (maxSteps < 1) {
      throw new AiError('PlannerExecutor maxSteps must be >= 1', 'invalid-request', {
        retryable: false,
      });
    }

    const tools = options?.tools ? resolveToolCallbacks(...options.tools) : undefined;
    const rounds: PlannerRound[] = [];
    const seen = new Set<string>();

    let plan: PlannerPlan =
      options?.initialPlan ?? (await this.plan(goal, undefined, undefined, options));

    rounds.push({ index: 0, phase: 'plan', plan });
    seen.add(planFingerprint(plan));

    let stepCount = 0;

    while (!plan.done) {
      throwIfAborted(options?.signal);
      stepCount += 1;
      if (stepCount > maxSteps) {
        throw new AiError(
          `PlannerExecutor exceeded maxSteps (${maxSteps}) without completing the goal`,
          'invalid-request',
          { retryable: false },
        );
      }

      const next = plan.steps.find((s) => s.status === 'pending');
      if (!next) {
        // No pending work but not done — force a replan that must finish or add steps.
        plan = await this.plan(
          goal,
          plan,
          'No pending steps remain but done=false. Finish or add steps.',
          options,
        );
        rounds.push({ index: rounds.length, phase: 'replan', plan });
        this.assertNoCycle(seen, plan);
        continue;
      }

      const observation = await this.act(goal, plan, next, tools, options);
      const afterAct: PlannerPlan = {
        ...plan,
        steps: plan.steps.map((s) =>
          s.id === next.id ? { ...s, status: 'done' as const, result: observation } : s,
        ),
      };
      rounds.push({
        index: rounds.length,
        phase: 'act',
        plan: afterAct,
        action: { stepId: next.id, observation },
      });

      plan = await this.plan(goal, afterAct, observation, options);
      rounds.push({ index: rounds.length, phase: 'replan', plan });
      this.assertNoCycle(seen, plan);
    }

    const answer =
      plan.finalAnswer?.trim() ||
      plan.steps
        .filter((s) => s.status === 'done' && s.result)
        .map((s) => s.result)
        .join('\n') ||
      plan.goal;

    return { answer, plan, rounds, stepCount };
  }

  private assertNoCycle(seen: Set<string>, plan: PlannerPlan): void {
    const fp = planFingerprint(plan);
    if (seen.has(fp)) {
      throw new AiError(
        'PlannerExecutor detected a repeated plan state (cycle protection)',
        'invalid-request',
        { retryable: false },
      );
    }
    seen.add(fp);
  }

  private async plan(
    goal: string,
    prior: PlannerPlan | undefined,
    observation: string | undefined,
    options?: PlannerExecutorOptions,
  ): Promise<PlannerPlan> {
    const userParts = [
      `Goal: ${goal}`,
      prior ? `Current plan:\n${JSON.stringify(prior, null, 2)}` : 'No prior plan.',
      observation ? `Latest observation:\n${observation}` : undefined,
      'Return the next plan JSON.',
    ].filter(Boolean);

    return callChatEntity<PlannerPlan>(this.chatClient, {
      system: options?.plannerSystem ?? DEFAULT_PLANNER,
      user: userParts.join('\n\n'),
      schema: PLAN_SCHEMA as unknown as Record<string, unknown>,
      signal: options?.signal,
      options: options?.options,
      map: (v) => normalizePlan(v, goal),
    });
  }

  private async act(
    goal: string,
    plan: PlannerPlan,
    step: PlannerStep,
    tools: ReturnType<typeof resolveToolCallbacks> | undefined,
    options?: PlannerExecutorOptions,
  ): Promise<string> {
    throwIfAborted(options?.signal);
    const user = [
      `Overall goal: ${goal}`,
      `Plan reasoning: ${plan.reasoning ?? '(none)'}`,
      `Execute this step only:`,
      `- id: ${step.id}`,
      `- description: ${step.description}`,
      step.toolName ? `- preferred tool: ${step.toolName}` : undefined,
      'Respond with a short observation of the outcome.',
    ]
      .filter(Boolean)
      .join('\n');

    if (tools?.length) {
      let spec = this.chatClient.prompt();
      spec = spec
        .system(options?.actorSystem ?? DEFAULT_ACTOR)
        .user(user)
        .tools(...tools);
      const merged: ChatOptions = {
        ...options?.options,
        signal: options?.signal ?? options?.options?.signal,
      };
      if (merged.signal !== undefined || options?.options) {
        spec = spec.options(merged);
      }
      return (await spec.call().content()) ?? '';
    }

    return callChatContent(this.chatClient, {
      system: options?.actorSystem ?? DEFAULT_ACTOR,
      user,
      signal: options?.signal,
      options: options?.options,
    });
  }
}

export function plannerExecutorWorkflow(chatClient: ChatClient): PlannerExecutorWorkflow {
  return PlannerExecutorWorkflow.of(chatClient);
}
