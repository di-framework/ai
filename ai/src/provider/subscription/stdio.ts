import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  CallToolResultSchema,
  ListToolsRequestSchema,
  ListToolsResultSchema,
} from '@modelcontextprotocol/sdk/types.js';

// Provider starts this process. Only MCP protocol messages may go to stdout.
const url = process.env.DI_BRIDGE_URL;
const token = process.env.DI_BRIDGE_TOKEN;
if (!url || !/^http:\/\/127\.0\.0\.1:\d+$/.test(url) || !token) {
  throw new Error('Missing local bridge connection');
}
async function rpc(path: string, body: unknown, signal?: AbortSignal) {
  const response = await fetch(`${url}${path}`, {
    method: 'POST',
    headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw new Error(`Tool bridge HTTP ${response.status}`);
  return response.json();
}
const server = new Server(
  { name: 'di-framework-tool-bridge', version: '0.1.0' },
  { capabilities: { tools: {} } },
);
const sessionId = crypto.randomUUID();
server.setRequestHandler(ListToolsRequestSchema, async (_request, extra) =>
  ListToolsResultSchema.parse(await rpc('/list', {}, extra.signal)),
);
server.setRequestHandler(CallToolRequestSchema, async (request, extra) =>
  CallToolResultSchema.parse(
    await rpc(
      '/call',
      {
        id: `${sessionId}:${extra.requestId}`,
        name: request.params.name,
        arguments: request.params.arguments ?? {},
      },
      extra.signal,
    ),
  ),
);
await server.connect(new StdioServerTransport());
// Some CLIs leave an MCP child alive after exiting. Stop once its owner disappears.
const watchdog = setInterval(async () => {
  try {
    const response = await fetch(`${url}/health`, {
      headers: { authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(1500),
    });
    if (response.ok) return;
  } catch {
    /* parent exited */
  }
  await server.close();
  process.exit(0);
}, 2000);
watchdog.unref();
process.stdin.on('end', () => {
  clearInterval(watchdog);
  void server.close();
});
