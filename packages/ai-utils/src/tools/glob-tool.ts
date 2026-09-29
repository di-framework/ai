import { statSync } from 'node:fs';
import { relative } from 'node:path';
import { functionToolCallback, type ToolCallback } from '@di-framework/ai';
import type { AiIgnorePolicy, AiIgnoreSuppressionDiagnostic } from '../policy/index.ts';
import {
  type AllowedDirectories,
  resolveAllowedDirectories,
} from '../sandbox/allowed-directories.ts';
import { assertPathAllowed, uniqueResolvedRoots } from '../sandbox/paths.ts';
import { compileGlob } from './glob-match.ts';
import { walkFiles } from './walk-files.ts';

const DEFAULT_MAX_RESULTS = 200;
const DEFAULT_MAX_DEPTH = 20;

export interface GlobToolOptions {
  readonly allowedDirectories: AllowedDirectories;
  readonly workingDirectory?: string;
  readonly maxResults?: number;
  readonly maxDepth?: number;
  readonly aiIgnorePolicy?: AiIgnorePolicy;
  readonly onSuppressed?: (diagnostic: AiIgnoreSuppressionDiagnostic) => void;
}

export interface GlobInput {
  readonly pattern?: string;
  readonly path?: string;
}

export function globTool(options: GlobToolOptions): ToolCallback {
  const maxResults = options.maxResults ?? DEFAULT_MAX_RESULTS;
  const maxDepth = options.maxDepth ?? DEFAULT_MAX_DEPTH;
  const defaultRoot = options.workingDirectory
    ? uniqueResolvedRoots([options.workingDirectory])[0]
    : undefined;

  return functionToolCallback<GlobInput, string>({
    name: 'Glob',
    description: `Fast file pattern matching under the allowed skill/workspace directories.
Supports globs like "**/*.md" or "scripts/*". Returns matching paths sorted by name.`,
    inputSchema: {
      type: 'object',
      properties: {
        pattern: {
          type: 'string',
          description: 'Glob pattern to match (e.g. "**/*.md")',
        },
        path: {
          type: 'string',
          description: 'Directory to search. Defaults to the workspace root.',
        },
      },
      required: ['pattern'],
    },
    call: (input) => {
      const pattern = input?.pattern?.trim() ?? '';
      if (!pattern) return 'Error: The glob pattern must not be empty';

      const roots = uniqueResolvedRoots(resolveAllowedDirectories(options.allowedDirectories));
      const searchRoot = input?.path?.trim() || defaultRoot || roots[0];
      if (!searchRoot) return 'Error: No search directory configured';

      const access = assertPathAllowed(searchRoot, roots);
      if (!access.ok) return access.error;

      try {
        if (!statSync(access.path).isDirectory()) {
          return `Error: Path is not a directory: ${searchRoot}`;
        }
      } catch {
        return `Error: Directory does not exist: ${searchRoot}`;
      }

      const matcher = compileGlob(pattern);
      const matches: string[] = [];
      walkFiles(
        access.path,
        0,
        maxDepth,
        (file) => {
          const rel = relative(access.path, file).split('\\').join('/');
          if (matcher(rel) || matcher(file.split('\\').join('/'))) {
            matches.push(file);
          }
        },
        {
          aiIgnorePolicy: options.aiIgnorePolicy,
          surface: 'glob',
          onSuppressed: options.onSuppressed,
        },
      );
      matches.sort((a, b) => a.localeCompare(b));
      const clipped = matches.slice(0, maxResults);
      if (clipped.length === 0) {
        return `No files matched "${pattern}" under ${access.path}`;
      }
      const suffix =
        matches.length > clipped.length ? `\n… ${matches.length - clipped.length} more` : '';
      return clipped.join('\n') + suffix;
    },
  });
}
