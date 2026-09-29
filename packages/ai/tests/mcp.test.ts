import { describe, expect, test } from 'bun:test';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import * as z from 'zod';
import {
  adaptSdkClient,
  ChatClient,
  createMcpToolCallbackProvider,
  defaultMcpToolNamePrefixGenerator,
  functionToolCallback,
  type McpCallToolParams,
  type McpCallToolResult,
  type McpClientSession,
  type McpListToolsResult,
  McpToolCallback,
  McpToolCallbackProvider,
  type McpToolDescriptor,
  mcpResultToString,
  mcpToolCallbacks,
  noPrefixMcpToolNameGenerator,
  prefixedToolName,
  ScriptedChatModel,
  ToolContext,
  ToolExecutionException,
  toolCall,
  toolCallbackAsMcpTool,
  toolCallResponse,
} from '../src/index.ts';
import { mcpToolCallback } from '../src/mcp/mcp-tool-callback.ts';
import { contentBlocksToString, emptyConnectionInfo } from '../src/mcp/mcp-tool-utils.ts';

class FakeMcpSession implements McpClientSession {
  readonly connectionInfo;
  readonly calls: McpCallToolParams[] = [];
  private tools: McpToolDescriptor[];
  private handlers: Record<
    string,
    (args: Record<string, unknown>) => McpCallToolResult | Promise<McpCallToolResult>
  >;

  constructor(options: {
    tools: McpToolDescriptor[];
    handlers: Record<
      string,
      (args: Record<string, unknown>) => McpCallToolResult | Promise<McpCallToolResult>
    >;
    connectionInfo?: McpClientSession['connectionInfo'];
  }) {
    this.tools = options.tools;
    this.handlers = options.handlers;
    this.connectionInfo = options.connectionInfo ?? {
      clientName: 'test-client',
      serverName: 'fake-server',
      title: 'demo',
    };
  }

  async listTools(): Promise<McpListToolsResult> {
    return { tools: this.tools };
  }

  async callTool(params: McpCallToolParams): Promise<McpCallToolResult> {
    this.calls.push(params);
    const handler = this.handlers[params.name];
    if (!handler) {
      return {
        isError: true,
        content: [{ type: 'text', text: `unknown tool ${params.name}` }],
      };
    }
    return handler(params.arguments ?? {});
  }
}

describe('MCP tool utils', () => {
  test('prefixedToolName formats and truncates', () => {
    const name = prefixedToolName('my-server', 'fs', 'read_file');
    expect(name).toContain('read_file');
    expect(name.length).toBeLessThanOrEqual(64);
  });

  test('prefixedToolName throws for empty prefix or toolName', () => {
    expect(() => prefixedToolName('', undefined, 'tool')).toThrow(/cannot be null or empty/);
    expect(() => prefixedToolName('prefix', undefined, '   ')).toThrow(/cannot be null or empty/);
  });

  test('prefixedToolName keeps the last 64 chars when the combined name overflows', () => {
    const longToolName = 'a'.repeat(80);
    const name = prefixedToolName('server', undefined, longToolName);
    expect(name.length).toBe(64);
    expect(name).toBe(`server_${longToolName}`.slice(-64));
  });

  test('emptyConnectionInfo returns an empty object', () => {
    expect(emptyConnectionInfo()).toEqual({});
  });

  test('contentBlocksToString serializes multiple blocks and non-text blocks', () => {
    expect(contentBlocksToString([])).toBe('');
    expect(
      contentBlocksToString([
        { type: 'text', text: 'a' },
        { type: 'text', text: 'b' },
      ]),
    ).toBe('[{"type":"text","text":"a"},{"type":"text","text":"b"}]');
    expect(contentBlocksToString([{ type: 'image', data: 'xyz' } as never])).toBe(
      '{"type":"image","data":"xyz"}',
    );
  });

  test('default prefix generator uses server name', () => {
    const name = defaultMcpToolNamePrefixGenerator(
      { serverName: 'weather', title: 'prod' },
      { name: 'get_forecast' },
    );
    expect(name).toMatch(/get_forecast$/);
    expect(name).toContain('weather');
  });

  test('noPrefix returns original name', () => {
    expect(noPrefixMcpToolNameGenerator({ serverName: 'x' }, { name: 't' })).toBe('t');
  });

  test('mcpResultToString prefers structuredContent', () => {
    expect(
      mcpResultToString({
        content: [{ type: 'text', text: 'ignored' }],
        structuredContent: { a: 1 },
      }),
    ).toBe('{"a":1}');
    expect(
      mcpResultToString({
        content: [{ type: 'text', text: 'hello' }],
      }),
    ).toBe('hello');
  });
});

describe('McpToolCallback', () => {
  test('calls MCP tool with original name and returns text', async () => {
    const session = new FakeMcpSession({
      tools: [
        {
          name: 'add',
          description: 'Add numbers',
          inputSchema: {
            type: 'object',
            properties: {
              a: { type: 'number' },
              b: { type: 'number' },
            },
            required: ['a', 'b'],
          },
        },
      ],
      handlers: {
        add: (args) => ({
          content: [
            {
              type: 'text',
              text: String(Number(args.a) + Number(args.b)),
            },
          ],
        }),
      },
    });

    const cb = new McpToolCallback({
      mcpClient: session,
      tool: (await session.listTools()).tools[0]!,
      prefixedToolName: 'fake_demo_add',
    });

    expect(cb.toolDefinition.name).toBe('fake_demo_add');
    expect(cb.originalToolName).toBe('add');
    const result = await cb.call('{"a":2,"b":3}');
    expect(result).toBe('5');
    expect(session.calls[0]?.name).toBe('add');
  });

  test('forwards tool context as _meta', async () => {
    const session = new FakeMcpSession({
      tools: [{ name: 'echo', inputSchema: { type: 'object', properties: {} } }],
      handlers: {
        echo: () => ({ content: [{ type: 'text', text: 'ok' }] }),
      },
    });
    const cb = new McpToolCallback({
      mcpClient: session,
      tool: { name: 'echo' },
    });
    await cb.call('{}', new ToolContext({ userId: 'u-1' }));
    expect(session.calls[0]?._meta).toEqual({ userId: 'u-1' });
  });

  test('isError results become ToolExecutionException', async () => {
    const session = new FakeMcpSession({
      tools: [{ name: 'boom' }],
      handlers: {
        boom: () => ({
          isError: true,
          content: [{ type: 'text', text: 'noop' }],
        }),
      },
    });
    const cb = new McpToolCallback({
      mcpClient: session,
      tool: { name: 'boom' },
    });
    await expect(cb.call('{}')).rejects.toBeInstanceOf(ToolExecutionException);
  });

  test('invalid JSON arguments throw ToolExecutionException', async () => {
    const session = new FakeMcpSession({
      tools: [{ name: 't' }],
      handlers: { t: () => ({ content: [] }) },
    });
    const cb = new McpToolCallback({
      mcpClient: session,
      tool: { name: 't' },
    });
    await expect(cb.call('not-json')).rejects.toBeInstanceOf(ToolExecutionException);
  });

  test('mcpToolCallback() factory constructs an equivalent McpToolCallback', async () => {
    const session = new FakeMcpSession({
      tools: [{ name: 'echo' }],
      handlers: { echo: () => ({ content: [{ type: 'text', text: 'ok' }] }) },
    });
    const cb = mcpToolCallback({ mcpClient: session, tool: { name: 'echo' } });
    expect(cb).toBeInstanceOf(McpToolCallback);
    expect(await cb.call('{}')).toBe('ok');
  });
});

describe('McpToolCallbackProvider', () => {
  test('refresh discovers tools; getToolCallbacks requires refresh', async () => {
    const session = new FakeMcpSession({
      tools: [
        {
          name: 'ping',
          description: 'ping',
          inputSchema: { type: 'object', properties: {} },
        },
      ],
      handlers: {
        ping: () => ({ content: [{ type: 'text', text: 'pong' }] }),
      },
    });

    const provider = new McpToolCallbackProvider({
      mcpClients: [session],
      toolNamePrefixGenerator: noPrefixMcpToolNameGenerator,
    });

    expect(() => provider.getToolCallbacks()).toThrow(/refresh/);

    await provider.refresh();
    const tools = provider.getToolCallbacks();
    expect(tools).toHaveLength(1);
    expect(tools[0]?.toolDefinition.name).toBe('ping');
    expect(await tools[0]?.call('{}')).toBe('pong');
  });

  test('invalidate clears the cache and requires another refresh', async () => {
    const session = new FakeMcpSession({
      tools: [{ name: 'ping' }],
      handlers: { ping: () => ({ content: [{ type: 'text', text: 'pong' }] }) },
    });
    const provider = new McpToolCallbackProvider({
      mcpClients: [session],
      toolNamePrefixGenerator: noPrefixMcpToolNameGenerator,
    });
    await provider.refresh();
    expect(provider.getToolCallbacks()).toHaveLength(1);

    provider.invalidate();
    expect(() => provider.getToolCallbacks()).toThrow(/refresh/);

    await provider.refresh();
    expect(provider.getToolCallbacks()).toHaveLength(1);
  });

  test('tool filter excludes tools', async () => {
    const session = new FakeMcpSession({
      tools: [{ name: 'keep' }, { name: 'drop' }],
      handlers: {
        keep: () => ({ content: [{ type: 'text', text: 'k' }] }),
        drop: () => ({ content: [{ type: 'text', text: 'd' }] }),
      },
    });
    const provider = await createMcpToolCallbackProvider({
      mcpClients: [session],
      toolFilter: (_c, tool) => tool.name === 'keep',
      toolNamePrefixGenerator: noPrefixMcpToolNameGenerator,
    });
    expect(provider.getToolCallbacks().map((t) => t.toolDefinition.name)).toEqual(['keep']);
  });

  test('duplicate names after prefix throw on refresh', async () => {
    const a = new FakeMcpSession({
      tools: [{ name: 'same' }],
      handlers: { same: () => ({ content: [{ type: 'text', text: 'a' }] }) },
      connectionInfo: { serverName: 's', title: 't' },
    });
    const b = new FakeMcpSession({
      tools: [{ name: 'same' }],
      handlers: { same: () => ({ content: [{ type: 'text', text: 'b' }] }) },
      connectionInfo: { serverName: 's', title: 't' },
    });
    const provider = new McpToolCallbackProvider({
      mcpClients: [a, b],
    });
    await expect(provider.refresh()).rejects.toThrow(/same name/);
  });

  test('mcpToolCallbacks convenience + ChatClient tool loop', async () => {
    const session = new FakeMcpSession({
      tools: [
        {
          name: 'getWeather',
          description: 'weather',
          inputSchema: {
            type: 'object',
            properties: { city: { type: 'string' } },
            required: ['city'],
          },
        },
      ],
      handlers: {
        getWeather: (args) => ({
          content: [
            {
              type: 'text',
              text: JSON.stringify({ temp: 68, city: args.city }),
            },
          ],
        }),
      },
      connectionInfo: { serverName: 'wx' },
    });

    const tools = await mcpToolCallbacks(session);
    const toolName = tools[0]?.toolDefinition.name;
    expect(toolName).toBeDefined();
    const model = new ScriptedChatModel([
      {
        respond: toolCallResponse([toolCall('c1', toolName!, { city: 'Yorktown' })]),
      },
      { respond: '68F in Yorktown' },
    ]);

    const answer = await ChatClient.create(model)
      .prompt()
      .user('weather in Yorktown?')
      .tools(...tools)
      .call()
      .content();

    expect(answer).toBe('68F in Yorktown');
    expect(session.calls[0]?.arguments).toEqual({ city: 'Yorktown' });
  });
});

describe('toolCallbackAsMcpTool (local → MCP)', () => {
  test('wraps FunctionToolCallback as MCP descriptor + handler', async () => {
    const local = functionToolCallback({
      name: 'double',
      description: 'double a number',
      inputSchema: {
        type: 'object',
        properties: { n: { type: 'number' } },
        required: ['n'],
      },
      call: ({ n }: { n: number }) => ({ result: n * 2 }),
    });
    const { descriptor, handler } = toolCallbackAsMcpTool(local);
    expect(descriptor.name).toBe('double');
    const result = await handler({ n: 21 });
    expect(result.isError).toBeUndefined();
    expect(result.content[0]).toMatchObject({ type: 'text' });
    expect(JSON.parse((result.content[0] as { text: string }).text)).toEqual({
      result: 42,
    });
  });
});

describe('adaptSdkClient', () => {
  test('normalizes missing tool result content without dropping extension fields', async () => {
    const session = adaptSdkClient({
      async listTools() {
        return { tools: [] };
      },
      async callTool() {
        return { content: undefined, requestId: 'request-1' };
      },
    });

    const result = await session.callTool({ name: 'test' });

    expect(result.content).toEqual([]);
    expect(result.requestId).toBe('request-1');
  });

  test('throws when passed a client without a listTools function', () => {
    expect(() => adaptSdkClient(undefined as never)).toThrow(/listTools\/callTool/);
    expect(() => adaptSdkClient({} as never)).toThrow(/listTools\/callTool/);
  });
});

describe('Official SDK InMemory transport + adaptSdkClient', () => {
  test('discovers and calls tools through McpToolCallbackProvider', async () => {
    const server = new McpServer({ name: 'memory-server', version: '1.0.0' });
    server.registerTool(
      'add',
      {
        description: 'Add two numbers',
        inputSchema: {
          // MCP SDK typings still expect Zod 3; runtime accepts Zod 4.
          a: z.number() as any,
          b: z.number() as any,
        },
      },
      async ({ a, b }: { a: number; b: number }) => ({
        content: [{ type: 'text', text: String(a + b) }],
      }),
    );

    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();

    await Promise.all([
      server.connect(serverTransport),
      (async () => {
        const client = new Client({ name: 'test-client', version: '1.0.0' });
        await client.connect(clientTransport);

        const session = adaptSdkClient(client, { title: 'mem' });
        const provider = await createMcpToolCallbackProvider({
          mcpClients: [session],
          toolNamePrefixGenerator: noPrefixMcpToolNameGenerator,
        });

        const tools = provider.getToolCallbacks();
        expect(tools.map((t) => t.toolDefinition.name)).toContain('add');

        const add = tools.find(
          (t): t is McpToolCallback => t instanceof McpToolCallback && t.originalToolName === 'add',
        )!;
        expect(await add.call(JSON.stringify({ a: 10, b: 5 }))).toBe('15');

        await client.close();
      })(),
    ]);

    await server.close();
  });
});
