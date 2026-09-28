import {
  ChatAgent,
  type ChatModel,
  functionToolCallback,
  type ToolCallback,
} from '@di-framework/ai';

const BLOCKED_FOR_SUBAGENTS = new Set(['Task', 'AskUserQuestion']);

export interface TaskToolOptions {
  readonly chatModel: ChatModel;
  readonly tools: readonly ToolCallback[];
  readonly system?: string;
  readonly maxTurnsHint?: number;
}

export interface TaskInput {
  readonly prompt?: string;
  readonly description?: string;
}

export function taskTool(options: TaskToolOptions): ToolCallback {
  const childTools = options.tools.filter(
    (tool) => !BLOCKED_FOR_SUBAGENTS.has(tool.toolDefinition.name),
  );

  return functionToolCallback<TaskInput, string>({
    name: 'Task',
    description: `Run a focused subagent on a bounded task and return its final answer.

The subagent cannot ask the user questions or spawn further tasks. Give a complete prompt.`,
    inputSchema: {
      type: 'object',
      properties: {
        prompt: { type: 'string', description: 'Full task for the subagent' },
        description: { type: 'string', description: 'Short label' },
      },
      required: ['prompt'],
    },
    call: async (input) => {
      const prompt = input?.prompt?.trim() ?? '';
      if (!prompt) return 'Error: prompt is required';
      const agent = ChatAgent.create({
        chatModel: options.chatModel,
        system:
          options.system ??
          'You are a focused subagent. Complete the assigned task and return a concise result. Do not ask the user questions.',
        tools: childTools,
      });
      const { content } = await agent.chat(prompt);
      return content;
    },
  });
}
