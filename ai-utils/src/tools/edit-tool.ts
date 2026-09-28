import { readFileSync, writeFileSync } from 'node:fs';
import { functionToolCallback, type ToolCallback } from '@di-framework/ai';
import {
  type AllowedDirectories,
  resolveAllowedDirectories,
} from '../sandbox/allowed-directories.ts';
import { errorMessage, nodeErrnoCode } from '../sandbox/fs-error.ts';
import { assertPathAllowed } from '../sandbox/paths.ts';
import { type AiIgnoreToolPolicy, aiIgnoreRejection } from './aiignore-enforcement.ts';
import { DEFAULT_MAX_LINE_CHARS } from './read-tool.ts';

export interface EditToolOptions {
  readonly allowedDirectories: AllowedDirectories;
  readonly aiIgnore?: AiIgnoreToolPolicy;
}

export interface EditInput {
  readonly filePath?: string;
  readonly oldString?: string;
  readonly newString?: string;
  readonly replaceAll?: boolean;
}

export function editTool(options: EditToolOptions): ToolCallback {
  return functionToolCallback<EditInput, string>({
    name: 'Edit',
    description: `Performs an exact string replacement in a file inside the allowed directories.

Usage:
- oldString must match the file text exactly (not Read line-number prefixes)
- If oldString appears more than once, either widen the context or set replaceAll
- newString must differ from oldString
- Read the file first so indentation is exact`,
    inputSchema: {
      type: 'object',
      properties: {
        filePath: {
          type: 'string',
          description: 'The path to edit (absolute preferred)',
        },
        oldString: {
          type: 'string',
          description: 'Exact text to replace',
        },
        newString: {
          type: 'string',
          description: 'Replacement text',
        },
        replaceAll: {
          type: 'boolean',
          description: 'Replace every occurrence (default false)',
        },
      },
      required: ['filePath', 'oldString', 'newString'],
    },
    call: (input) => {
      const filePath = input?.filePath?.trim() ?? '';
      const oldString = input?.oldString ?? '';
      const newString = input?.newString ?? '';
      const access = assertPathAllowed(
        filePath,
        resolveAllowedDirectories(options.allowedDirectories),
      );
      if (!access.ok) return access.error;
      const ignored = aiIgnoreRejection(options.aiIgnore, access.path, 'read', 'file');
      if (ignored != null) return ignored;
      if (oldString === newString) {
        return 'Error: oldString and newString must be different';
      }

      let content: string;
      try {
        content = readFileSync(access.path, 'utf8');
      } catch (error) {
        const code = nodeErrnoCode(error);
        if (code === 'ENOENT') {
          return `Error: File does not exist: ${filePath}`;
        }
        if (code === 'EISDIR') {
          return `Error: Path is a directory, not a file: ${filePath}`;
        }
        return `Error reading file: ${errorMessage(error)}`;
      }

      const occurrences = countOccurrences(content, oldString);
      if (occurrences === 0) {
        return `Error: oldString was not found in ${filePath}`;
      }
      if (occurrences > 1 && !input?.replaceAll) {
        return `Error: oldString appears ${occurrences} times in the file. Either provide a larger string with more surrounding context to make it unique or use replaceAll=true to change all instances.`;
      }

      const next = input?.replaceAll
        ? content.split(oldString).join(newString)
        : content.replace(oldString, newString);

      try {
        writeFileSync(access.path, next, 'utf8');
      } catch (error) {
        return `Error writing file: ${errorMessage(error)}`;
      }

      const snippet = numberedSnippet(next, newString);
      return `The file ${access.path} has been updated. Here's the result of running \`cat -n\` on a snippet of the edited file:\n${snippet}`;
    },
  });
}

function countOccurrences(haystack: string, needle: string): number {
  if (needle.length === 0) return 0;
  let count = 0;
  let from = 0;
  while (from <= haystack.length) {
    const index = haystack.indexOf(needle, from);
    if (index === -1) break;
    count += 1;
    from = index + needle.length;
  }
  return count;
}

function numberedSnippet(content: string, focus: string): string {
  const lines = content.split(/\r?\n/);
  if (lines.length > 0 && lines[lines.length - 1] === '') {
    lines.pop();
  }
  const focusLine = content.indexOf(focus);
  let start = 0;
  if (focusLine >= 0) {
    const before = content.slice(0, focusLine);
    const lineIndex = before.split(/\r?\n/).length - 1;
    start = Math.max(0, lineIndex - 2);
  }
  const end = Math.min(lines.length, start + 7);
  return lines
    .slice(start, end)
    .map((line, index) => {
      const text =
        line.length > DEFAULT_MAX_LINE_CHARS
          ? `${line.slice(0, DEFAULT_MAX_LINE_CHARS)}... (line truncated)`
          : line;
      const n = start + index + 1;
      return `${String(n).padStart(6, ' ')}\t${text}`;
    })
    .join('\n');
}
