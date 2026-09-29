import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, URL } from 'node:url';
import { ChatResponse } from '../../chat/model/chat-response.ts';
import type { Prompt } from '../../chat/prompt/prompt.ts';
import { AiError } from '../../model/errors.ts';
import { BRIDGE_NAME, createBridgeLaunch } from './launch.ts';
import type { SubscriptionChatModelOptions } from './model.ts';
import { startToolBridge } from './server.ts';

export async function callSubscription(
  config: SubscriptionChatModelOptions,
  prompt: Prompt,
): Promise<ChatResponse> {
  for (const [key, value] of Object.entries(prompt.options ?? {})) {
    if (value == null || ['signal', 'model', 'toolCallbacks', 'toolContext'].includes(key))
      continue;
    if (key === 'providerOptions' && Object.keys(value).length === 0) continue;
    throw new AiError(`MCP CLI bridge does not support ChatOptions.${key}`, 'invalid-request');
  }
  for (const message of prompt.messages) {
    if ('media' in message && message.media.length)
      throw new AiError('MCP CLI bridge supports text only', 'invalid-request');
  }
  const timeoutMs = config.timeoutMs ?? 120_000;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1)
    throw new AiError('timeoutMs must be positive', 'invalid-request');
  const lifetime = new AbortController();
  const signal = AbortSignal.any([
    lifetime.signal,
    AbortSignal.timeout(timeoutMs),
    ...(prompt.options?.signal ? [prompt.options.signal] : []),
  ]);
  signal.throwIfAborted();
  const executable =
    config.executable ?? Bun.which(config.provider, { PATH: config.env?.PATH ?? process.env.PATH });
  if (!executable)
    throw new AiError(`Install and sign in to the ${config.provider} CLI`, 'authentication');
  const callPrompt = prompt.withOptions({
    signal,
    toolContext: { ...prompt.options?.toolContext, signal },
  });
  const bridge = startToolBridge({
    prompt: callPrompt,
    signal,
    manager: config.toolCallingManager,
    maxCalls: config.maxCalls,
    onEvent: config.onEvent,
  });
  let directory: string | undefined;
  let child: ReturnType<typeof Bun.spawn> | undefined;
  const kill = () => {
    child?.kill('SIGKILL');
  };
  signal.addEventListener('abort', kill, { once: true });
  try {
    directory = await mkdtemp(join(tmpdir(), 'di-mcp-bridge-'));
    // Providers discover project customizations relative to a real repository root.
    const init = Bun.spawn(['git', 'init', '--quiet', directory], {
      stdout: 'ignore',
      stderr: 'pipe',
    });
    if ((await init.exited) !== 0)
      throw new Error('Could not initialize the temporary bridge workspace');
    const input = [
      `You are running the caller's agent. Its tools are supplied by MCP server ${BRIDGE_NAME}.`,
      'Use that server for the listed application tools, not similarly named built-in tools.',
      'Do not modify files or run shell commands. The conversation below contains the caller instructions and task.',
      'Answer normally after using the relevant MCP tools; do not print tool requests as JSON.',
      JSON.stringify(
        prompt.messages.map((message) => ({
          role: message.messageType,
          text: message.text,
          ...('toolCalls' in message ? { toolCalls: message.toolCalls } : {}),
          ...('responses' in message ? { responses: message.responses } : {}),
        })),
      ),
    ].join('\n\n');
    const args = await createBridgeLaunch(
      config.provider,
      directory,
      {
        command: process.execPath,
        args: [fileURLToPath(new URL('./stdio.ts', import.meta.url))],
        env: { DI_BRIDGE_URL: bridge.url, DI_BRIDGE_TOKEN: bridge.token },
      },
      input,
      prompt.options?.model ?? config.model,
    );
    signal.throwIfAborted();
    const processHandle = Bun.spawn([executable, ...args], {
      cwd: directory,
      env: { ...process.env, ...config.env },
      stdin: 'ignore',
      stdout: 'pipe',
      stderr: 'inherit',
    });
    child = processHandle;
    if (signal.aborted) kill();
    const readOutput = async () => {
      const reader = processHandle.stdout.getReader();
      const cancelRead = () => {
        void reader.cancel().catch(() => {});
      };
      signal.addEventListener('abort', cancelRead, { once: true });
      if (signal.aborted) cancelRead();
      const chunks: Uint8Array[] = [];
      let bytes = 0;
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          bytes += value.byteLength;
          if (bytes > 4 * 1024 * 1024) throw new Error('CLI output exceeded 4 MiB');
          chunks.push(value);
        }
      } finally {
        signal.removeEventListener('abort', cancelRead);
        reader.releaseLock();
      }
      return Buffer.concat(chunks).toString('utf8').trim();
    };
    const [text, exitCode] = await Promise.all([readOutput(), processHandle.exited]);
    signal.throwIfAborted();
    if (exitCode !== 0) throw new Error(`${config.provider} exited ${exitCode}; see stderr`);
    if (prompt.options?.toolCallbacks?.length && !bridge.connected)
      throw new Error(
        `${config.provider} did not connect to the MCP tool bridge. CLI response: ${text.slice(0, 2000)}`,
      );
    if (!text) throw new Error(`${config.provider} returned no text`);
    // MCP already executed the calls: returning native ToolCalls here would execute them twice.
    return ChatResponse.of(text, { bridgeEvents: bridge.events, provider: config.provider });
  } catch (cause) {
    if (signal.aborted) {
      throw new AiError(
        prompt.options?.signal?.aborted
          ? 'Subscription request cancelled'
          : 'Subscription request timed out',
        prompt.options?.signal?.aborted ? 'cancelled' : 'timeout',
        { provider: config.provider, cause, retryable: false },
      );
    }
    throw cause;
  } finally {
    lifetime.abort();
    kill();
    signal.removeEventListener('abort', kill);
    if (child) await child.exited;
    await bridge.close();
    if (directory) await rm(directory, { recursive: true, force: true });
  }
}
