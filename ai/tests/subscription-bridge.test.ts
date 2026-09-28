import { afterEach, expect, test } from 'bun:test';
import { chmod, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { createToolCallingManager, functionToolCallback, Prompt } from '../src/index.ts';
import { BRIDGE_NAME, bridgeProviders } from '../src/provider/subscription/launch.ts';
import { SubscriptionChatModel as CliToolChatModel } from '../src/provider/subscription/model.ts';
import { type BridgeEvent, startToolBridge } from '../src/provider/subscription/server.ts';

const directories: string[] = [];
afterEach(async () => {
  await Promise.all(
    directories.splice(0).map((path) => rm(path, { recursive: true, force: true })),
  );
});
async function temporary() {
  const path = await mkdtemp(join(tmpdir(), 'di-bridge-test-'));
  directories.push(path);
  return path;
}
async function fakeProvider(source: string) {
  const path = join(await temporary(), 'fake-provider');
  await writeFile(path, `#!${process.execPath}\n${source}`);
  await chmod(path, 0o755);
  return path;
}
const sdkImports = `
import { Client } from ${JSON.stringify(import.meta.resolve('@modelcontextprotocol/sdk/client/index.js'))};
import { StdioClientTransport } from ${JSON.stringify(import.meta.resolve('@modelcontextprotocol/sdk/client/stdio.js'))};
`;

for (const provider of bridgeProviders) {
  test(`${provider}: native MCP process executes callbacks exactly once`, async () => {
    const log = join(await temporary(), 'launch.json');
    const skillPath = 'host-skill.md';
    const executable = await fakeProvider(`${sdkImports}
      const args = process.argv.slice(2);
      let server;
      if (${JSON.stringify(provider)} === 'codex') {
        const override = args.find(a => a.startsWith('mcp_servers.'));
        server = Bun.TOML.parse(override).mcp_servers[${JSON.stringify(BRIDGE_NAME)}];
      } else if (${JSON.stringify(provider)} === 'grok') {
        server = Bun.TOML.parse(await Bun.file('.grok/config.toml').text()).mcp_servers[${JSON.stringify(BRIDGE_NAME)}];
      } else {
        const path = ${JSON.stringify(provider)} === 'claude' ? args[args.indexOf('--mcp-config') + 1]
          : ${JSON.stringify(provider)} === 'agy' ? '.agents/mcp_config.json'
          : ${JSON.stringify(provider)} === 'junie' ? '.junie/mcp/mcp.json' : '.mcp.json';
        server = (await Bun.file(path).json()).mcpServers[${JSON.stringify(BRIDGE_NAME)}];
      }
      
      await Bun.write(${JSON.stringify(log)}, JSON.stringify({cwd: process.cwd(), url: server.env.DI_BRIDGE_URL}));
      const client = new Client({name:'fake-provider',version:'1'});
      await client.connect(new StdioClientTransport({...server, stderr: 'inherit'}));
      try {
        const listed = await client.listTools();
        if (listed.tools.length !== 2) throw new Error('Wrong tool catalog');
        const skill = await client.callTool({name:'Skill',arguments:{command:'explain-injection'}});
        if (skill.isError || !JSON.stringify(skill).includes('constructor injection')) throw new Error('Skill failed');
        const read = await client.callTool({name:'Read',arguments:{filePath:${JSON.stringify(skillPath)},limit:3}});
        if (read.isError || !JSON.stringify(read).includes('name: explain-injection')) throw new Error('Read failed');
        console.log('MCP round trip completed');
      } finally { await client.close(); }
    `);
    const events: BridgeEvent[] = [];
    const model = new CliToolChatModel({
      provider,
      executable,
      onEvent: (event) => events.push(event),
      timeoutMs: 20_000,
    });
    const response = await model.call(
      new Prompt('Explain injection.', {
        toolCallbacks: [
          functionToolCallback({ name: 'Skill', call: () => 'constructor injection' }),
          functionToolCallback({ name: 'Read', call: () => 'name: explain-injection' }),
        ],
      }),
    );
    expect(response.hasToolCalls()).toBe(false);
    const result = response.result?.output.text;
    expect(result).toBe('MCP round trip completed');
    expect(events.map((event) => event.type)).toEqual([
      'tool_use',
      'tool_result',
      'tool_use',
      'tool_result',
    ]);
    expect(events.filter((event) => event.type === 'tool_use').map((event) => event.name)).toEqual([
      'Skill',
      'Read',
    ]);
    const launch = JSON.parse(await readFile(log, 'utf8'));
    expect(await Bun.file(join(launch.cwd, 'mcp.json')).exists()).toBe(false);
    await expect(readFile(launch.cwd)).rejects.toThrow();
    await expect(fetch(`${launch.url}/health`)).rejects.toThrow();
  }, 30_000);
}

test('MCP client sees errors and context stays in the host; configured advisors execute', async () => {
  const invocations: unknown[] = [];
  let advisorCalls = 0;
  const callback = functionToolCallback<{ value: string }, string>({
    name: 'Echo',
    inputSchema: { type: 'object', properties: { value: { type: 'string' } }, required: ['value'] },
    call: (input, context) => {
      invocations.push(context?.get('secret'));
      return input.value;
    },
  });
  const controller = new AbortController();
  const bridge = startToolBridge({
    signal: controller.signal,
    prompt: new Prompt('test', { toolCallbacks: [callback], toolContext: { secret: 'host-only' } }),
    manager: createToolCallingManager({
      advisors: [
        {
          name: 'test-advisor',
          order: 0,
          async adviseExecution(context, chain) {
            advisorCalls++;
            return chain(context);
          },
        },
      ],
    }),
  });
  const client = new Client({ name: 'test-client', version: '1' });
  try {
    await client.connect(
      new StdioClientTransport({
        command: process.execPath,
        args: [join(import.meta.dir, '../src/provider/subscription/stdio.ts')],
        env: { DI_BRIDGE_URL: bridge.url, DI_BRIDGE_TOKEN: bridge.token },
        stderr: 'inherit',
      }),
    );
    const catalog = await client.listTools();
    expect(JSON.stringify(catalog)).not.toContain('host-only');
    expect((await client.callTool({ name: 'Missing', arguments: {} })).isError).toBe(true);
    expect((await client.callTool({ name: 'Echo', arguments: { value: 42 } })).isError).toBe(true);
    const result = await client.callTool({
      name: 'Echo',
      arguments: { value: 'literal $(whoami)' },
    });
    expect(result.content).toEqual([{ type: 'text', text: 'literal $(whoami)' }]);
    expect(invocations).toEqual(['host-only']);
    expect(advisorCalls).toBe(1);
    expect((await fetch(`${bridge.url}/list`, { method: 'POST' })).status).toBe(401);
  } finally {
    await client.close();
    controller.abort();
    await bridge.close();
  }
});

test('invalid JSON, invalid calls, and tool execution errors return bridge failures', async () => {
  const bridge = startToolBridge({
    signal: new AbortController().signal,
    prompt: new Prompt('test', {
      toolCallbacks: [
        functionToolCallback({
          name: 'Boom',
          call: () => {
            throw new Error('callback exploded');
          },
        }),
      ],
    }),
  });
  const headers = { authorization: `Bearer ${bridge.token}` };
  try {
    const invalidJson = await fetch(`${bridge.url}/call`, {
      method: 'POST',
      headers,
      body: '{not-json',
    });
    expect(invalidJson.status).toBe(400);
    expect(await invalidJson.text()).toBe('Invalid JSON');

    const invalidCall = await fetch(`${bridge.url}/call`, {
      method: 'POST',
      headers: { ...headers, 'content-type': 'application/json' },
      body: JSON.stringify({ id: 'one', name: 'Boom', arguments: ['not-an-object'] }),
    });
    expect(invalidCall.status).toBe(400);
    expect(await invalidCall.text()).toBe('Invalid call');

    const executionError = await fetch(`${bridge.url}/call`, {
      method: 'POST',
      headers: { ...headers, 'content-type': 'application/json' },
      body: JSON.stringify({ id: 'two', name: 'Boom', arguments: {} }),
    });
    expect(executionError.status).toBe(200);
    expect(await executionError.json()).toEqual({
      isError: true,
      content: [{ type: 'text', text: "Tool 'Boom' failed: callback exploded" }],
    });
    expect(bridge.events).toEqual([
      { type: 'tool_use', id: 'two', name: 'Boom', input: {} },
      {
        type: 'tool_result',
        tool_use_id: 'two',
        content: "Tool 'Boom' failed: callback exploded",
        is_error: true,
      },
    ]);
  } finally {
    await bridge.close();
  }
});

test('replayed request IDs are idempotent and tool call limits apply', async () => {
  let calls = 0;
  const callback = functionToolCallback({ name: 'Count', call: () => String(++calls) });
  const bridge = startToolBridge({
    signal: new AbortController().signal,
    maxCalls: 1,
    prompt: new Prompt('test', { toolCallbacks: [callback] }),
  });
  const call = (id: string) =>
    fetch(`${bridge.url}/call`, {
      method: 'POST',
      headers: { authorization: `Bearer ${bridge.token}` },
      body: JSON.stringify({ id, name: 'Count', arguments: {} }),
    }).then((response) => response.json());
  try {
    const results = await Promise.all([call('one'), call('one')]);
    expect(results[0]).toEqual(results[1]);
    expect(calls).toBe(1);
    expect(await call('two')).toMatchObject({ isError: true });
    expect(calls).toBe(1);
  } finally {
    await bridge.close();
  }
});

test('nonzero exit, missing MCP connection, and cancellation cannot look successful', async () => {
  const exit = await fakeProvider('process.exit(7)');
  await expect(
    new CliToolChatModel({ provider: 'codex', executable: exit }).call(new Prompt('test')),
  ).rejects.toThrow('exited 7');
  const disconnected = await fakeProvider('console.log("fake answer")');
  await expect(
    new CliToolChatModel({ provider: 'codex', executable: disconnected }).call(
      new Prompt('test', {
        toolCallbacks: [functionToolCallback({ name: 'Echo', call: () => 'ok' })],
      }),
    ),
  ).rejects.toThrow('did not connect');
  const hung = await fakeProvider('setInterval(() => {}, 1000)');
  const abort = new AbortController();
  const task = new CliToolChatModel({ provider: 'codex', executable: hung }).call(
    new Prompt('test', { signal: abort.signal }),
  );
  setTimeout(() => abort.abort(), 100);
  await expect(task).rejects.toThrow();
});

test('plain text does not require MCP discovery; timeout and cancellation use AiError codes', async () => {
  const executable = await fakeProvider('console.log("plain answer")');
  const model = new CliToolChatModel({ provider: 'codex', executable });
  expect((await model.call(new Prompt('hello'))).result?.output.text).toBe('plain answer');
  await expect(model.call(new Prompt('hello', { temperature: 0.2 }))).rejects.toMatchObject({
    code: 'invalid-request',
  });
  const hung = await fakeProvider('setInterval(() => {},1000)');
  await expect(
    new CliToolChatModel({ provider: 'codex', executable: hung, timeoutMs: 100 }).call(
      new Prompt('hello'),
    ),
  ).rejects.toMatchObject({ code: 'timeout' });
  await expect(
    model.call(new Prompt('hello', { signal: AbortSignal.abort() })),
  ).rejects.toMatchObject({ code: 'cancelled' });
});

for (const allowed of [true, false]) {
  test(`ToolCallingAdvisor manager authorizes subscription callbacks: ${allowed}`, async () => {
    const { ChatClient, ToolCallingAdvisor, toolAuthorizationAdvisor } = await import(
      '../src/index.ts'
    );
    let executions = 0;
    let authorizations = 0;
    const executable = await fakeProvider(`${sdkImports}
      const server = Bun.TOML.parse(process.argv.find(a => a.startsWith('mcp_servers.'))).mcp_servers[${JSON.stringify(BRIDGE_NAME)}];
      const client = new Client({name:'authorization-test',version:'1'});
      await client.connect(new StdioClientTransport({...server, stderr:'inherit'}));
      await client.listTools();
      const result = await client.callTool({name:'Count',arguments:{}});
      console.log(JSON.stringify(result));
      await client.close();
    `);
    const manager = createToolCallingManager({
      advisors: [
        toolAuthorizationAdvisor({
          authorizationManager: {
            authorize(principal) {
              authorizations++;
              expect(principal?.sub).toBe('trusted-user');
              return { allowed };
            },
          },
        }),
      ],
    });
    const model = new CliToolChatModel({
      provider: 'codex',
      executable,
      toolCallingManager: {
        resolveToolDefinitions: () => [],
        executeToolCalls: async () => {
          throw new Error('Wrong manager');
        },
      },
    });
    const client = ChatClient.builder(model)
      .defaultAdvisors(new ToolCallingAdvisor({ toolCallingManager: manager }))
      .build();
    const content = await client
      .prompt()
      .user('count')
      .options({
        toolCallbacks: [
          functionToolCallback({
            name: 'Count',
            call: () => {
              executions++;
              return 'counted';
            },
          }),
        ],
        toolContext: { principal: { sub: 'trusted-user', method: 'bearer' } },
      })
      .call()
      .content();
    expect(authorizations).toBe(1);
    expect(executions).toBe(allowed ? 1 : 0);
    expect(content).toContain(allowed ? 'counted' : 'unauthorized');
  });
}

test('concurrent tool calls execute sequentially and cancelled queued calls never execute', async () => {
  const controller = new AbortController();
  let release: (() => void) | undefined;
  let started: (() => void) | undefined;
  const entered = new Promise<void>((resolve) => {
    started = resolve;
  });
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  let calls = 0;
  const bridge = startToolBridge({
    signal: controller.signal,
    prompt: new Prompt('test', {
      toolCallbacks: [
        functionToolCallback({
          name: 'Wait',
          call: async () => {
            calls++;
            started?.();
            await pending;
            return 'finished';
          },
        }),
      ],
    }),
  });
  const call = (id: string) =>
    fetch(`${bridge.url}/call`, {
      method: 'POST',
      headers: { authorization: `Bearer ${bridge.token}` },
      body: JSON.stringify({ id, name: 'Wait', arguments: {} }),
    }).then((response) => (response.ok ? response.json() : { status: response.status }));
  try {
    const first = call('first');
    await entered;
    const second = call('second');
    controller.abort();
    release?.();
    const results = await Promise.all([first, second]);
    expect(calls).toBe(1);
    // A queued call returns an MCP cancellation; a call arriving after abort gets HTTP 410.
    expect(results[0]).toMatchObject({ content: [{ type: 'text', text: 'finished' }] });
    expect(results[1]).toBeDefined();
  } finally {
    release?.();
    await bridge.close();
  }
});

test('stdout overflow fails and closes the process', async () => {
  const executable = await fakeProvider(
    'console.log("x".repeat(4 * 1024 * 1024 + 1)); setInterval(() => {},1000)',
  );
  await expect(
    new CliToolChatModel({ provider: 'codex', executable }).call(new Prompt('test')),
  ).rejects.toMatchObject({ code: 'provider-error', message: 'CLI output exceeded 4 MiB' });
});
