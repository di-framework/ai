import {
  AiError,
  assistantMessage,
  ChatResponse,
  createToolCallingManager,
  type McpCallToolResult,
  type Prompt,
  type ToolCallingManager,
  toolCall,
  toolCallbackToMcpDescriptor,
  validateAgainstJsonSchema,
} from '../../index.ts';

// Wire names match Terella's cli-common ContentBlock variants.
export type BridgeEvent =
  | { type: 'tool_use'; id: string; name: string; input: Record<string, unknown> }
  | { type: 'tool_result'; tool_use_id: string; content: string; is_error?: boolean };

export function startToolBridge(options: {
  prompt: Prompt;
  signal: AbortSignal;
  manager?: ToolCallingManager;
  onEvent?: (event: BridgeEvent) => void;
  maxCalls?: number;
}) {
  const callbacks = options.prompt.options?.toolCallbacks ?? [];
  const tools = new Map(callbacks.map((tool) => [tool.toolDefinition.name, tool]));
  if (tools.size !== callbacks.length)
    throw new AiError('Duplicate bridge tool names', 'invalid-request');
  for (const tool of callbacks) {
    if (tool.toolMetadata?.returnDirect)
      throw new AiError('MCP bridge does not support returnDirect tools', 'invalid-request');
  }
  const maxCalls = options.maxCalls ?? 32;
  if (!Number.isSafeInteger(maxCalls) || maxCalls < 1)
    throw new AiError('maxCalls must be positive', 'invalid-request');
  const manager = options.manager ?? createToolCallingManager();
  const token = crypto.randomUUID();
  const events: BridgeEvent[] = [];
  const requests = new Map<string, { signature: string; result: Promise<McpCallToolResult> }>();
  let queue: Promise<unknown> = Promise.resolve();
  let connected = false;
  const emit = (event: BridgeEvent) => {
    events.push(event);
    // Observers must not turn a completed side effect into a retryable failure.
    try {
      options.onEvent?.(event);
    } catch {
      /* observer errors do not alter execution */
    }
  };
  const error = (message: string): McpCallToolResult => ({
    isError: true,
    content: [{ type: 'text', text: message }],
  });
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    maxRequestBodySize: 1024 * 1024,
    async fetch(request) {
      if (
        request.headers.get('authorization') !== `Bearer ${token}` ||
        request.headers.has('origin')
      ) {
        return new Response('Unauthorized', { status: 401 });
      }
      if (options.signal.aborted) return new Response('Bridge closed', { status: 410 });
      const path = new URL(request.url).pathname;
      if (path === '/health' && request.method === 'GET') return new Response('ok');
      if (request.method !== 'POST') return new Response('Method not allowed', { status: 405 });
      if (path === '/list') {
        connected = true;
        return Response.json({ tools: callbacks.map(toolCallbackToMcpDescriptor) });
      }
      if (path !== '/call') return new Response('Not found', { status: 404 });
      let body: unknown;
      try {
        body = await request.json();
      } catch {
        return new Response('Invalid JSON', { status: 400 });
      }
      if (!body || typeof body !== 'object' || Array.isArray(body))
        return new Response('Invalid call', { status: 400 });
      const { id, name, arguments: args } = body as Record<string, unknown>;
      if (
        typeof id !== 'string' ||
        id.length > 200 ||
        !id ||
        typeof name !== 'string' ||
        !args ||
        typeof args !== 'object' ||
        Array.isArray(args)
      ) {
        return new Response('Invalid call', { status: 400 });
      }
      const tool = tools.get(name);
      if (!tool) return Response.json(error(`Unknown tool: ${name}`));
      const validation = validateAgainstJsonSchema(args, tool.toolDefinition.inputSchema);
      if (!validation.success)
        return Response.json(error(`Invalid arguments: ${validation.errorMessage}`));
      const signature = JSON.stringify({ name, args });
      const existing = requests.get(id);
      if (existing) {
        if (existing.signature !== signature)
          return new Response('Reused call ID', { status: 409 });
        return Response.json(await existing.result);
      }
      if (requests.size >= maxCalls)
        return Response.json(error(`Tool call limit (${maxCalls}) exceeded`));
      const execute = async (): Promise<McpCallToolResult> => {
        if (options.signal.aborted) return error('Tool call cancelled');
        emit({ type: 'tool_use', id, name, input: args as Record<string, unknown> });
        let result: McpCallToolResult;
        try {
          const response = ChatResponse.fromAssistant(
            assistantMessage('', {
              toolCalls: [toolCall(id, name, args as Record<string, unknown>)],
            }),
          );
          // Keep callback instances, tool context, guards and execution advisors in this process.
          const execution = await manager.executeToolCalls(options.prompt, response);
          const last = execution.conversationHistory.at(-1);
          if (last?.messageType !== 'tool')
            throw new Error('Tool manager returned no tool response');
          result = {
            content: last.responses.map((item) => ({
              type: 'text' as const,
              text: item.responseData,
            })),
          };
        } catch (cause) {
          result = error(cause instanceof Error ? cause.message : 'Tool execution failed');
        }
        emit({
          type: 'tool_result',
          tool_use_id: id,
          content:
            result.content?.map((block) => (block.type === 'text' ? block.text : '')).join('\n') ??
            '',
          ...(result.isError ? { is_error: true } : {}),
        });
        return result;
      };
      // Serial execution preserves ai-utils activation/guard state even if MCP calls arrive together.
      const result = queue.then(execute, execute);
      queue = result;
      requests.set(id, { signature, result });
      return Response.json(await result);
    },
  });
  return {
    url: `http://127.0.0.1:${server.port}`,
    token,
    events,
    get connected() {
      return connected;
    },
    async close() {
      await server.stop(true);
    },
  };
}
