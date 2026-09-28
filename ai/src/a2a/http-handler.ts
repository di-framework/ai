import { AgentCardHelper } from './agent-card.ts';
import type { A2AAgentExecutor } from './executor.ts';
import { A2AJsonRpcHandler } from './jsonrpc.ts';
import { A2ATaskStore } from './task-store.ts';
import { AGENT_CARD_WELL_KNOWN_PATH, type AgentCard } from './types.ts';

export interface A2AHttpHandlerOptions {
  readonly card: AgentCard;
  readonly taskStore?: A2ATaskStore;
  readonly executor?: A2AAgentExecutor;
  readonly jsonRpcHandler?: A2AJsonRpcHandler;
  readonly basePath?: string;
  readonly authHandler?: (
    request: Request,
  ) => Promise<boolean | Response | null | undefined> | boolean | Response | null | undefined;
}

/**
 * Creates a standard Fetch (Request -> Response) HTTP handler that serves:
 * 1. GET /.well-known/agent-card.json
 * 2. POST (JSON-RPC endpoint) for SendMessage, GetTask, ListTasks, CancelTask
 *
 * Fully portable across Bun, Node.js (fetch), Deno, and Cloudflare Workers.
 */
function trimTrailingSlashes(str: string): string {
  let end = str.length;
  while (end > 0 && str.charCodeAt(end - 1) === 47 /* '/' */) {
    end--;
  }
  return str.slice(0, end);
}

export function createA2AHttpHandler(
  options: A2AHttpHandlerOptions,
): (request: Request) => Promise<Response> {
  const card = options.card;
  const taskStore = options.taskStore ?? A2ATaskStore.create();
  const jsonRpcHandler =
    options.jsonRpcHandler ??
    A2AJsonRpcHandler.create({
      taskStore,
      executor: options.executor,
    });

  const rawBasePath = trimTrailingSlashes(options.basePath ?? '');
  const wellKnownPath = `${rawBasePath}${AGENT_CARD_WELL_KNOWN_PATH}`;
  const shortCardPath = `${rawBasePath}/agent-card.json`;

  return async (request: Request): Promise<Response> => {
    const url = new URL(request.url, 'http://localhost');
    const pathname = trimTrailingSlashes(url.pathname) || '/';

    // 1. GET Agent Card
    if (request.method === 'GET') {
      if (
        pathname === AGENT_CARD_WELL_KNOWN_PATH ||
        pathname === wellKnownPath ||
        pathname === shortCardPath ||
        pathname.endsWith('/.well-known/agent-card.json') ||
        pathname.endsWith('/agent-card.json')
      ) {
        return new Response(AgentCardHelper.serialize(card), {
          status: 200,
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': 'public, max-age=300',
          },
        });
      }

      return new Response(
        JSON.stringify({
          error: 'Not Found',
          message: `Endpoint '${pathname}' not found. Use GET ${AGENT_CARD_WELL_KNOWN_PATH}`,
        }),
        {
          status: 404,
          headers: { 'Content-Type': 'application/json' },
        },
      );
    }

    // 2. Auth hook if configured
    if (options.authHandler) {
      const authResult = await options.authHandler(request);
      if (authResult instanceof Response) {
        return authResult;
      }
      if (authResult === false) {
        return new Response(
          JSON.stringify({
            jsonrpc: '2.0',
            id: null,
            error: {
              code: -32001,
              message: 'Unauthorized: invalid or missing credentials',
            },
          }),
          {
            status: 401,
            headers: { 'Content-Type': 'application/json' },
          },
        );
      }
    }

    // 3. POST JSON-RPC endpoint
    if (request.method === 'POST') {
      return jsonRpcHandler.handleHttpRequest(request);
    }

    // 4. Method not allowed
    return new Response(
      JSON.stringify({
        error: 'Method Not Allowed',
        message: `HTTP method '${request.method}' is not supported`,
      }),
      {
        status: 405,
        headers: {
          'Content-Type': 'application/json',
          Allow: 'GET, POST',
        },
      },
    );
  };
}
