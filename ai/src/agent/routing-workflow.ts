import type { ChatClient } from '../chat/client/default-chat-client.ts';
import { AiError } from '../model/errors.ts';
import { callChatContent, callChatEntity, type WorkflowCallOptions } from './workflow-utils.ts';

/**
 * A route target: static specialist system prompt, dedicated ChatClient, or custom handler.
 */
export type RouteHandler =
  | string
  | ChatClient
  | ((input: string, options?: WorkflowCallOptions) => string | Promise<string>);

export type RouteMap = Readonly<Record<string, RouteHandler>>;

export interface RoutingWorkflowResult {
  readonly route: string;
  readonly reasoning?: string;
  readonly result: string;
}

export interface RoutingWorkflowOptions extends WorkflowCallOptions {
  /**
   * Fallback route key when classification is invalid.
   * Defaults to the first key in the route map.
   */
  readonly defaultRoute?: string;
  /**
   * Optional classification system prompt override.
   */
  readonly classificationSystem?: string;
}

const ROUTE_SCHEMA = {
  type: 'object',
  properties: {
    route: { type: 'string' },
    reasoning: { type: 'string' },
  },
  required: ['route'],
} as const;

/**
 * Classify input into a route, then run the specialized handler for that route.
 * Spring AI / Anthropic: Routing Workflow.
 *
 * @example
 * ```ts
 * const result = await new RoutingWorkflow(chatClient).route(
 *   "My account was charged twice",
 *   {
 *     billing: "You are a billing specialist…",
 *     technical: "You are a technical support engineer…",
 *     general: "You are a customer service representative…",
 *   },
 * );
 * ```
 */
export class RoutingWorkflow {
  private readonly chatClient: ChatClient;

  constructor(chatClient: ChatClient) {
    this.chatClient = chatClient;
  }

  /** Factory alias (same as {@link routingWorkflow}). */
  static of(chatClient: ChatClient): RoutingWorkflow {
    return new RoutingWorkflow(chatClient);
  }

  async route(input: string, routes: RouteMap, options?: RoutingWorkflowOptions): Promise<string> {
    const detailed = await this.routeDetailed(input, routes, options);
    return detailed.result;
  }

  async routeDetailed(
    input: string,
    routes: RouteMap,
    options?: RoutingWorkflowOptions,
  ): Promise<RoutingWorkflowResult> {
    const keys = Object.keys(routes);
    if (keys.length === 0) {
      throw new Error('RoutingWorkflow requires at least one route');
    }

    const classificationSystem =
      options?.classificationSystem ??
      [
        'You are a request router.',
        'Classify the user message into exactly one of the following routes:',
        keys.map((k) => `- ${k}`).join('\n'),
        'Respond with JSON only matching the schema {"route": string, "reasoning"?: string}.',
        'The route value must be one of the listed route names.',
      ].join('\n');

    const classified = await callChatEntity<{
      route: string;
      reasoning?: string;
    }>(this.chatClient, {
      system: classificationSystem,
      user: input,
      schema: ROUTE_SCHEMA as unknown as Record<string, unknown>,
      signal: options?.signal,
      options: options?.options,
    });

    let routeKey = classified.route?.trim();
    if (!routeKey || !(routeKey in routes)) {
      const fallback = options?.defaultRoute ?? keys[0]!;
      if (!(fallback in routes)) {
        throw new AiError(
          `Invalid route "${routeKey}" and fallback "${fallback}" not in routes`,
          'invalid-request',
          { retryable: false },
        );
      }
      routeKey = fallback;
    }

    const handler = routes[routeKey]!;
    const result = await invokeRoute(handler, input, options, this.chatClient);

    return {
      route: routeKey,
      reasoning: classified.reasoning,
      result,
    };
  }
}

async function invokeRoute(
  handler: RouteHandler,
  input: string,
  options: RoutingWorkflowOptions | undefined,
  defaultClient: ChatClient,
): Promise<string> {
  if (typeof handler === 'function') {
    return handler(input, options);
  }
  if (typeof handler === 'string') {
    return callChatContent(defaultClient, {
      system: handler,
      user: input,
      signal: options?.signal,
      options: options?.options,
    });
  }
  // Dedicated ChatClient for this route
  return callChatContent(handler, {
    user: input,
    signal: options?.signal,
    options: options?.options,
  });
}

export function routingWorkflow(chatClient: ChatClient): RoutingWorkflow {
  return RoutingWorkflow.of(chatClient);
}
