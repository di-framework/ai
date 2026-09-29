import type { ChatResponse } from '../model/chat-response.ts';

/**
 * Response returned by the advisor chain, aligned with Spring AI
 * {@code ChatClientResponse}.
 */
export interface ChatClientResponse {
  readonly chatResponse: ChatResponse | undefined;
  readonly context: Map<string, unknown>;
}

export function chatClientResponse(
  chatResponse: ChatResponse | undefined,
  context: Map<string, unknown> | Readonly<Record<string, unknown>> = new Map(),
): ChatClientResponse {
  const map = context instanceof Map ? new Map(context) : new Map(Object.entries(context));
  return { chatResponse, context: map };
}

export function copyChatClientResponse(
  response: ChatClientResponse,
  overrides: {
    chatResponse?: ChatResponse | undefined;
    context?: Map<string, unknown> | Readonly<Record<string, unknown>>;
  } = {},
): ChatClientResponse {
  const context =
    overrides.context instanceof Map
      ? new Map(overrides.context)
      : overrides.context
        ? new Map([...response.context, ...Object.entries(overrides.context)])
        : new Map(response.context);

  return {
    chatResponse: 'chatResponse' in overrides ? overrides.chatResponse : response.chatResponse,
    context,
  };
}
