import { mkdirSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { functionToolCallback, type ToolCallback } from '@di-framework/ai';
import { nodeErrnoCode } from '../sandbox/fs-error.ts';
import { assertPathAllowed, expandUserPath, uniqueResolvedRoots } from '../sandbox/paths.ts';

export const MEMORY_SYSTEM_PROMPT = `You have a long-term memory directory at {MEMORIES_ROOT}.
Use MemoryView / MemoryWrite / MemoryEdit / MemoryDelete / MemoryRename for durable facts
(user preferences, project decisions, external references). Do not store secrets.
Prefer a MEMORY.md index that lists other memory files.`;

export interface MemoryToolsOptions {
  readonly directory: string;
}

export function memoryTools(options: MemoryToolsOptions): ToolCallback[] {
  const root = uniqueResolvedRoots([expandUserPath(options.directory)])[0];
  if (!root) throw new Error('Memory directory is required');
  mkdirSync(root, { recursive: true });
  const dirs = [root];

  const resolveMemory = (relPath: string): ReturnType<typeof assertPathAllowed> => {
    const trimmed = relPath.trim();
    if (!trimmed) return { ok: false, error: 'Error: path is required' };
    const target = trimmed.startsWith('/') ? trimmed : join(root, trimmed);
    return assertPathAllowed(target, dirs);
  };

  const view = functionToolCallback<{ path?: string }, string>({
    name: 'MemoryView',
    description: 'Read a memory file relative to the memories root, or list the root.',
    inputSchema: {
      type: 'object',
      properties: { path: { type: 'string', description: 'Relative path (omit to list root)' } },
    },
    call: (input) => {
      const rel = input?.path?.trim() ?? '';
      if (!rel) {
        return listTree(root);
      }
      const access = resolveMemory(rel);
      if (!access.ok) return access.error;
      try {
        return readFileSync(access.path, 'utf8');
      } catch (error) {
        if (nodeErrnoCode(error) === 'EISDIR') {
          return listTree(access.path);
        }
        return `Error: Memory does not exist: ${rel}`;
      }
    },
  });

  const write = functionToolCallback<{ path?: string; content?: string }, string>({
    name: 'MemoryWrite',
    description: 'Create or overwrite a memory markdown file under the memories root.',
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string' },
        content: { type: 'string' },
      },
      required: ['path'],
    },
    call: (input) => {
      const access = resolveMemory(input?.path ?? '');
      if (!access.ok) return access.error;
      mkdirSync(dirname(access.path), { recursive: true });
      writeFileSync(access.path, input?.content ?? '', 'utf8');
      return `Wrote ${relative(root, access.path) || access.path}`;
    },
  });

  const edit = functionToolCallback<
    { path?: string; oldString?: string; newString?: string },
    string
  >({
    name: 'MemoryEdit',
    description: 'Replace exact text in a memory file.',
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string' },
        oldString: { type: 'string' },
        newString: { type: 'string' },
      },
      required: ['path', 'oldString', 'newString'],
    },
    call: (input) => {
      const access = resolveMemory(input?.path ?? '');
      if (!access.ok) return access.error;
      let current: string;
      try {
        current = readFileSync(access.path, 'utf8');
      } catch {
        return `Error: Memory does not exist: ${input?.path}`;
      }
      const oldString = input?.oldString ?? '';
      if (!current.includes(oldString)) return 'Error: oldString was not found';
      writeFileSync(access.path, current.replace(oldString, input?.newString ?? ''), 'utf8');
      return `Updated ${relative(root, access.path)}`;
    },
  });

  const del = functionToolCallback<{ path?: string }, string>({
    name: 'MemoryDelete',
    description: 'Delete a memory file or empty directory under the memories root.',
    inputSchema: {
      type: 'object',
      properties: { path: { type: 'string' } },
      required: ['path'],
    },
    call: (input) => {
      const access = resolveMemory(input?.path ?? '');
      if (!access.ok) return access.error;
      try {
        rmSync(access.path, { recursive: false, force: false });
      } catch {
        return `Error: Memory does not exist: ${input?.path}`;
      }
      return `Deleted ${relative(root, access.path)}`;
    },
  });

  const rename = functionToolCallback<{ from?: string; to?: string }, string>({
    name: 'MemoryRename',
    description: 'Rename a memory file inside the memories root.',
    inputSchema: {
      type: 'object',
      properties: { from: { type: 'string' }, to: { type: 'string' } },
      required: ['from', 'to'],
    },
    call: (input) => {
      const from = resolveMemory(input?.from ?? '');
      const to = resolveMemory(input?.to ?? '');
      if (!from.ok) return from.error;
      if (!to.ok) return to.error;
      try {
        mkdirSync(dirname(to.path), { recursive: true });
        renameSync(from.path, to.path);
      } catch {
        return `Error: Memory does not exist: ${input?.from}`;
      }
      return `Renamed ${relative(root, from.path)} → ${relative(root, to.path)}`;
    },
  });

  return [view, write, edit, del, rename];
}

export function formatMemorySystemPrompt(directory: string): string {
  return MEMORY_SYSTEM_PROMPT.replace('{MEMORIES_ROOT}', directory);
}

function listTree(dir: string): string {
  const names = readdirSync(dir);
  if (names.length === 0) return `Empty memory directory: ${dir}`;
  return names.sort().join('\n');
}
