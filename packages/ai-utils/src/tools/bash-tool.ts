import { spawn } from 'node:child_process';
import { functionToolCallback, type ToolCallback } from '@di-framework/ai';
import {
  type AllowedDirectories,
  resolveAllowedDirectories,
} from '../sandbox/allowed-directories.ts';
import { assertPathAllowed, expandUserPath, uniqueResolvedRoots } from '../sandbox/paths.ts';

export const DEFAULT_BASH_TIMEOUT_MS = 120_000;
export const MAX_BASH_TIMEOUT_MS = 120_000;
const DEFAULT_MAX_OUTPUT_CHARS = 24_000;

export interface BashConfirmInput {
  readonly command: string;
  readonly cwd: string;
}

export interface BashToolOptions {
  readonly allowedDirectories: AllowedDirectories;
  readonly workingDirectory?: string;
  readonly timeoutMs?: number;
  readonly maxOutputChars?: number;
  /** Human-in-the-loop gate. Return false to reject the command. */
  readonly confirm?: (input: BashConfirmInput) => boolean | Promise<boolean>;
}

export interface BashInput {
  readonly command?: string;
  readonly timeout?: number;
  readonly description?: string;
  readonly cwd?: string;
}

/**
 * Run a shell command with cwd jailed to allowed roots.
 *
 * This is not a container sandbox: the process can still {@code cd} or open
 * the network. Only use on trusted hosts.
 */
export function bashTool(options: BashToolOptions): ToolCallback {
  const defaultTimeout = Math.min(
    options.timeoutMs ?? DEFAULT_BASH_TIMEOUT_MS,
    MAX_BASH_TIMEOUT_MS,
  );
  const maxOutputChars = options.maxOutputChars ?? DEFAULT_MAX_OUTPUT_CHARS;
  const defaultCwd = options.workingDirectory
    ? uniqueResolvedRoots([options.workingDirectory])[0]
    : undefined;

  return functionToolCallback<BashInput, string>({
    name: 'Bash',
    description: `Execute a shell command in the skill/workspace sandbox.

cwd is restricted to allowed directories (workspace + skill folders). This is not a full OS sandbox — the command can still reach the network or other paths if the host allows it. Prefer skill scripts under the skill base directory.

Usage:
- command is required
- optional timeout in milliseconds (max ${MAX_BASH_TIMEOUT_MS})
- optional cwd (must be inside allowed directories)`,
    inputSchema: {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'The shell command to run' },
        timeout: {
          type: 'integer',
          description: `Timeout in milliseconds (default ${defaultTimeout}, max ${MAX_BASH_TIMEOUT_MS})`,
        },
        description: {
          type: 'string',
          description: 'Short description of what the command does',
        },
        cwd: {
          type: 'string',
          description: 'Working directory. Defaults to the workspace root.',
        },
      },
      required: ['command'],
    },
    call: async (input) => {
      const command = input?.command?.trim() ?? '';
      if (!command) return 'Error: command is required';

      const roots = uniqueResolvedRoots(resolveAllowedDirectories(options.allowedDirectories));
      const cwdInput = input?.cwd?.trim() || defaultCwd || roots[0];
      if (!cwdInput) return 'Error: No working directory configured';

      const access = assertPathAllowed(cwdInput, roots);
      if (!access.ok) return access.error;

      if (options.confirm && !(await options.confirm({ command, cwd: access.path }))) {
        return `Error: Command was not approved: ${command}`;
      }

      const timeout = Math.min(Math.max(1, input?.timeout ?? defaultTimeout), MAX_BASH_TIMEOUT_MS);

      return runCommand(command, access.path, timeout, maxOutputChars);
    },
  });
}

function runCommand(
  command: string,
  cwd: string,
  timeoutMs: number,
  maxOutputChars: number,
): Promise<string> {
  const isWin = process.platform === 'win32';
  const file = isWin ? 'cmd.exe' : '/bin/sh';
  const args = isWin ? ['/d', '/s', '/c', command] : ['-c', command];

  return new Promise((resolvePromise) => {
    const child = spawn(file, args, {
      cwd: expandUserPath(cwd),
      env: process.env,
      windowsHide: true,
    });

    let stdout = '';
    let stderr = '';
    let settled = false;

    const finish = (text: string) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolvePromise(text);
    };

    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      finish(`Error: Command timed out after ${timeoutMs}ms`);
    }, timeoutMs);

    child.stdout?.setEncoding('utf8');
    child.stderr?.setEncoding('utf8');
    child.stdout?.on('data', (chunk: string) => {
      stdout += chunk;
    });
    child.stderr?.on('data', (chunk: string) => {
      stderr += chunk;
    });
    child.on('error', (error) => {
      finish(`Error starting command: ${error.message}`);
    });
    child.on('close', (code, signal) => {
      const body = [
        `cwd: ${cwd}`,
        `exit: ${code ?? 'null'}${signal ? ` signal=${signal}` : ''}`,
        stdout ? `stdout:\n${truncate(stdout, maxOutputChars)}` : 'stdout: (empty)',
        stderr ? `stderr:\n${truncate(stderr, maxOutputChars)}` : 'stderr: (empty)',
      ].join('\n');
      finish(body);
    });
  });
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max)}\n… (truncated ${text.length - max} chars)`;
}
