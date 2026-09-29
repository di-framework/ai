import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { functionToolCallback, type ToolCallback } from '@di-framework/ai';
import {
  type AllowedDirectories,
  resolveAllowedDirectories,
} from '../sandbox/allowed-directories.ts';
import { errorMessage, nodeErrnoCode } from '../sandbox/fs-error.ts';
import { assertPathAllowed } from '../sandbox/paths.ts';
import { type AiIgnoreToolPolicy, aiIgnoreRejection } from './aiignore-enforcement.ts';

export interface WriteToolOptions {
  readonly allowedDirectories: AllowedDirectories;
  readonly aiIgnore?: AiIgnoreToolPolicy;
}

export interface WriteInput {
  readonly filePath?: string;
  readonly content?: string;
}

export function writeTool(options: WriteToolOptions): ToolCallback {
  return functionToolCallback<WriteInput, string>({
    name: 'Write',
    description: `Writes a file in the allowed directories (creates or overwrites).

Usage:
- Prefer Edit for existing files
- Read a file before overwriting it
- Parent directories are created automatically
- filePath should be absolute when possible`,
    inputSchema: {
      type: 'object',
      properties: {
        filePath: {
          type: 'string',
          description: 'The path to write (absolute preferred)',
        },
        content: {
          type: 'string',
          description: 'File contents. Defaults to empty.',
        },
      },
      required: ['filePath'],
    },
    call: (input) => {
      const filePath = input?.filePath?.trim() ?? '';
      const access = assertPathAllowed(
        filePath,
        resolveAllowedDirectories(options.allowedDirectories),
      );
      if (!access.ok) return access.error;
      const ignored = aiIgnoreRejection(options.aiIgnore, access.path, 'write', 'file');
      if (ignored != null) return ignored;

      const content = input?.content ?? '';
      try {
        mkdirSync(dirname(access.path), { recursive: true });
        writeFileSync(access.path, content, { encoding: 'utf8', flag: 'wx' });
        const bytes = Buffer.byteLength(content, 'utf8');
        return `Successfully created file: ${access.path} (${bytes} bytes)`;
      } catch (error) {
        if (nodeErrnoCode(error) === 'EEXIST') {
          try {
            writeFileSync(access.path, content, 'utf8');
            const bytes = Buffer.byteLength(content, 'utf8');
            return `Successfully overwrote file: ${access.path} (${bytes} bytes)`;
          } catch (overwriteError) {
            if (nodeErrnoCode(overwriteError) === 'EISDIR') {
              return `Error: Path is a directory, not a file: ${filePath}`;
            }
            return `Error writing file: ${errorMessage(overwriteError)}`;
          }
        }
        return `Error writing file: ${errorMessage(error)}`;
      }
    },
  });
}
