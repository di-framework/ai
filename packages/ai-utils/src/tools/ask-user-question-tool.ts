import { functionToolCallback, type ToolCallback } from '@di-framework/ai';

export interface AskUserOption {
  readonly label: string;
  readonly description?: string;
}

export interface AskUserQuestion {
  readonly question: string;
  readonly header?: string;
  readonly options?: readonly AskUserOption[];
  readonly multiSelect?: boolean;
}

export interface AskUserQuestionInput {
  readonly questions?: readonly AskUserQuestion[];
}

export type QuestionHandler = (
  questions: readonly AskUserQuestion[],
) =>
  | Record<string, string | readonly string[]>
  | Promise<Record<string, string | readonly string[]>>;

export interface AskUserQuestionToolOptions {
  readonly questionHandler: QuestionHandler;
}

export function askUserQuestionTool(options: AskUserQuestionToolOptions): ToolCallback {
  return functionToolCallback<AskUserQuestionInput, string>({
    name: 'AskUserQuestion',
    description: `Ask the user one or more clarifying questions before continuing.

Usage:
- Provide 1-10 questions
- Each question should be complete and end with ?
- Include 2-4 options when the choice is constrained
- multiSelect allows more than one option`,
    inputSchema: {
      type: 'object',
      properties: {
        questions: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              question: { type: 'string' },
              header: { type: 'string' },
              options: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    label: { type: 'string' },
                    description: { type: 'string' },
                  },
                  required: ['label'],
                },
              },
              multiSelect: { type: 'boolean' },
            },
            required: ['question'],
          },
        },
      },
      required: ['questions'],
    },
    call: async (input) => {
      const questions = input?.questions ?? [];
      if (questions.length === 0) return 'Error: questions must not be empty';
      if (questions.length > 10) return 'Error: at most 10 questions are allowed';
      for (const question of questions) {
        if (!question.question?.trim()) return 'Error: each question text is required';
      }
      const answers = await options.questionHandler(questions);
      return JSON.stringify(answers, null, 2);
    },
  });
}
