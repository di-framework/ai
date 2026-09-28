import type { Prompt } from '../prompt/prompt.ts';

/**
 * Request flowing through the advisor chain, aligned with Spring AI
 * {@code ChatClientRequest}.
 */
export interface ChatClientRequest {
  readonly prompt: Prompt;
  readonly context: Map<string, unknown>;
}

export function chatClientRequest(
  prompt: Prompt,
  context: Map<string, unknown> | Readonly<Record<string, unknown>> = new Map(),
): ChatClientRequest {
  const map = context instanceof Map ? new Map(context) : new Map(Object.entries(context));
  return { prompt, context: map };
}

export function copyChatClientRequest(
  request: ChatClientRequest,
  overrides: {
    prompt?: Prompt;
    context?: Map<string, unknown> | Readonly<Record<string, unknown>>;
  } = {},
): ChatClientRequest {
  const context =
    overrides.context instanceof Map
      ? new Map(overrides.context)
      : overrides.context
        ? new Map([...request.context, ...Object.entries(overrides.context)])
        : new Map(request.context);

  return {
    prompt: overrides.prompt ?? request.prompt,
    context,
  };
}
