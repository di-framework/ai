import type { ChatClient } from '../chat/client/default-chat-client.ts';
import { callChatContent, callChatEntity, type WorkflowCallOptions } from './workflow-utils.ts';

export interface EvaluationResponse {
  readonly pass: boolean;
  readonly feedback: string;
  readonly score?: number;
}

export interface GenerationRecord {
  readonly response: string;
  readonly evaluation?: EvaluationResponse;
}

export interface RefinedResponse {
  readonly solution: string;
  readonly chainOfThought: readonly GenerationRecord[];
  readonly iterations: number;
}

export interface EvaluatorOptimizerWorkflowOptions extends WorkflowCallOptions {
  /** Max generate→evaluate rounds. Default 5. */
  readonly maxIterations?: number;
  readonly generatorSystem?: string;
  readonly evaluatorSystem?: string;
}

const EVAL_SCHEMA = {
  type: 'object',
  properties: {
    pass: { type: 'boolean' },
    feedback: { type: 'string' },
    score: { type: 'number' },
  },
  required: ['pass', 'feedback'],
} as const;

const DEFAULT_GENERATOR = [
  "You generate a high-quality solution for the user's task.",
  'If feedback is provided, revise the previous solution to address it fully.',
].join('\n');

const DEFAULT_EVALUATOR = [
  'You evaluate a candidate solution against the original task.',
  'Return JSON only: {"pass": boolean, "feedback": string, "score"?: number}.',
  'pass=true only when the solution fully satisfies the task with no critical issues.',
  'feedback must explain what is wrong or confirm acceptance.',
].join('\n');

/**
 * Generate → evaluate → refine loop until the evaluator passes or max iterations.
 * Spring AI / Anthropic: Evaluator-Optimizer.
 */
export class EvaluatorOptimizerWorkflow {
  private readonly chatClient: ChatClient;

  constructor(chatClient: ChatClient) {
    this.chatClient = chatClient;
  }

  /** Factory alias (same as {@link evaluatorOptimizerWorkflow}). */
  static of(chatClient: ChatClient): EvaluatorOptimizerWorkflow {
    return new EvaluatorOptimizerWorkflow(chatClient);
  }

  async loop(task: string, options?: EvaluatorOptimizerWorkflowOptions): Promise<RefinedResponse> {
    const maxIterations = Math.max(1, options?.maxIterations ?? 5);
    const chainOfThought: GenerationRecord[] = [];
    let context = '';
    let lastSolution = '';

    for (let i = 0; i < maxIterations; i++) {
      const generatorUser = context
        ? [
            `Task:\n${task}`,
            `Previous attempt:\n${lastSolution}`,
            `Feedback to address:\n${context}`,
            'Produce an improved solution.',
          ].join('\n\n')
        : `Task:\n${task}\n\nProduce a solution.`;

      const response = await callChatContent(this.chatClient, {
        system: options?.generatorSystem ?? DEFAULT_GENERATOR,
        user: generatorUser,
        signal: options?.signal,
        options: options?.options,
      });
      lastSolution = response;

      const evaluation = await callChatEntity<EvaluationResponse>(this.chatClient, {
        system: options?.evaluatorSystem ?? DEFAULT_EVALUATOR,
        user: [
          `Task:\n${task}`,
          `Candidate solution:\n${response}`,
          'Evaluate the candidate.',
        ].join('\n\n'),
        schema: EVAL_SCHEMA as unknown as Record<string, unknown>,
        signal: options?.signal,
        options: options?.options,
        map: (v) => {
          const obj = v as EvaluationResponse;
          return {
            pass: Boolean(obj.pass),
            feedback: String(obj.feedback ?? ''),
            score: typeof obj.score === 'number' ? obj.score : undefined,
          };
        },
      });

      chainOfThought.push({ response, evaluation });

      if (evaluation.pass) {
        return {
          solution: response,
          chainOfThought,
          iterations: i + 1,
        };
      }
      context = evaluation.feedback;
    }

    return {
      solution: lastSolution,
      chainOfThought,
      iterations: chainOfThought.length,
    };
  }
}

export function evaluatorOptimizerWorkflow(chatClient: ChatClient): EvaluatorOptimizerWorkflow {
  return EvaluatorOptimizerWorkflow.of(chatClient);
}
