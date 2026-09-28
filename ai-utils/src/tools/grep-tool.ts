import { readFileSync, statSync } from 'node:fs';
import { relative } from 'node:path';
import { functionToolCallback, type ToolCallback } from '@di-framework/ai';
import {
  type AiIgnorePolicy,
  type AiIgnoreSuppressionDiagnostic,
  aiIgnoreSuppressionDiagnostic,
  evaluateAiIgnorePath,
} from '../policy/index.ts';
import {
  type AllowedDirectories,
  resolveAllowedDirectories,
} from '../sandbox/allowed-directories.ts';
import { assertPathAllowed, uniqueResolvedRoots } from '../sandbox/paths.ts';
import { compileGlob } from './glob-match.ts';
import { walkFiles } from './walk-files.ts';

export type GrepOutputMode = 'content' | 'files_with_matches' | 'count';

export interface GrepToolOptions {
  readonly allowedDirectories: AllowedDirectories;
  readonly workingDirectory?: string;
  readonly maxResults?: number;
  readonly maxDepth?: number;
  readonly maxLineChars?: number;
  readonly aiIgnorePolicy?: AiIgnorePolicy;
  readonly onSuppressed?: (diagnostic: AiIgnoreSuppressionDiagnostic) => void;
}

export interface GrepInput {
  readonly pattern?: string;
  readonly path?: string;
  readonly glob?: string;
  readonly outputMode?: GrepOutputMode;
  readonly caseInsensitive?: boolean;
  readonly headLimit?: number;
  readonly before?: number;
  readonly after?: number;
}

const DEFAULT_MAX_RESULTS = 200;
const DEFAULT_MAX_DEPTH = 20;
const DEFAULT_MAX_LINE_CHARS = 2000;

export function grepTool(options: GrepToolOptions): ToolCallback {
  const maxResults = options.maxResults ?? DEFAULT_MAX_RESULTS;
  const maxDepth = options.maxDepth ?? DEFAULT_MAX_DEPTH;
  const maxLineChars = options.maxLineChars ?? DEFAULT_MAX_LINE_CHARS;
  const defaultRoot = options.workingDirectory
    ? uniqueResolvedRoots([options.workingDirectory])[0]
    : undefined;

  return functionToolCallback<GrepInput, string>({
    name: 'Grep',
    description: `Search file contents under the allowed skill/workspace directories.

Usage:
- pattern is a JavaScript regular expression
- Optional glob filters files (e.g. "**/*.ts")
- outputMode: content (default), files_with_matches, or count
- Optional before/after context lines for content mode`,
    inputSchema: {
      type: 'object',
      properties: {
        pattern: {
          type: 'string',
          description: 'Regular expression to search for',
        },
        path: {
          type: 'string',
          description: 'File or directory to search. Defaults to the workspace root.',
        },
        glob: {
          type: 'string',
          description: 'Glob filter for file names (e.g. "**/*.md")',
        },
        outputMode: {
          type: 'string',
          enum: ['content', 'files_with_matches', 'count'],
          description: 'How to report matches (default content)',
        },
        caseInsensitive: {
          type: 'boolean',
          description: 'Case-insensitive search',
        },
        headLimit: {
          type: 'integer',
          description: `Maximum matches or files to return (default ${maxResults})`,
        },
        before: {
          type: 'integer',
          description: 'Context lines before each match (content mode)',
        },
        after: {
          type: 'integer',
          description: 'Context lines after each match (content mode)',
        },
      },
      required: ['pattern'],
    },
    call: (input) => {
      const pattern = input?.pattern ?? '';
      if (!pattern) return 'Error: The search pattern must not be empty';

      let regex: RegExp;
      try {
        regex = new RegExp(pattern, input?.caseInsensitive ? 'gi' : 'g');
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return `Error: Invalid regular expression: ${message}`;
      }

      const roots = uniqueResolvedRoots(resolveAllowedDirectories(options.allowedDirectories));
      const searchRoot = input?.path?.trim() || defaultRoot || roots[0];
      if (!searchRoot) return 'Error: No search directory configured';

      const access = assertPathAllowed(searchRoot, roots);
      if (!access.ok) return access.error;

      const files: string[] = [];
      let stat: ReturnType<typeof statSync>;
      try {
        stat = statSync(access.path);
      } catch {
        return `Error: Path does not exist: ${searchRoot}`;
      }
      if (stat.isFile()) {
        const evaluation = options.aiIgnorePolicy
          ? evaluateAiIgnorePath(options.aiIgnorePolicy, access.path, { kind: 'file' })
          : undefined;
        if (evaluation?.ignored) {
          options.onSuppressed?.(aiIgnoreSuppressionDiagnostic(evaluation, 'grep', 'file'));
        } else {
          files.push(access.path);
        }
      } else if (stat.isDirectory()) {
        walkFiles(access.path, 0, maxDepth, (file) => files.push(file), {
          aiIgnorePolicy: options.aiIgnorePolicy,
          surface: 'grep',
          onSuppressed: options.onSuppressed,
        });
      } else {
        return `Error: Path is not a file or directory: ${searchRoot}`;
      }

      const globFilter = input?.glob?.trim() ? compileGlob(input.glob.trim()) : undefined;
      const limit = Math.max(1, input?.headLimit ?? maxResults);
      const mode: GrepOutputMode = input?.outputMode ?? 'content';
      const before = Math.max(0, input?.before ?? 0);
      const after = Math.max(0, input?.after ?? 0);

      const fileHits: { file: string; count: number; lines: string[] }[] = [];
      for (const file of files) {
        const rel = relative(access.path, file).split('\\').join('/') || file;
        const skipGlob =
          globFilter != null && !globFilter(rel) && !globFilter(file.split('\\').join('/'));
        if (skipGlob) continue;
        let text: string;
        try {
          text = readFileSync(file, 'utf8');
        } catch {
          continue;
        }
        if (text.includes('\0')) continue;

        const lines = text.split(/\r?\n/);
        if (lines.length > 0 && lines[lines.length - 1] === '') {
          lines.pop();
        }
        const matchIndexes: number[] = [];
        for (let i = 0; i < lines.length; i += 1) {
          regex.lastIndex = 0;
          if (regex.test(lines[i] ?? '')) {
            matchIndexes.push(i);
          }
        }
        if (matchIndexes.length === 0) continue;

        const shown = new Set<number>();
        const rendered: string[] = [];
        for (const lineIndex of matchIndexes) {
          const from = Math.max(0, lineIndex - before);
          const to = Math.min(lines.length - 1, lineIndex + after);
          for (let n = from; n <= to; n += 1) {
            if (shown.has(n)) continue;
            shown.add(n);
            const raw = lines[n] ?? '';
            const textLine =
              raw.length > maxLineChars ? `${raw.slice(0, maxLineChars)}... (line truncated)` : raw;
            rendered.push(`${file}:${n + 1}:${textLine}`);
          }
        }
        fileHits.push({ file, count: matchIndexes.length, lines: rendered });
      }

      if (fileHits.length === 0) {
        return `No matches for /${pattern}/ under ${access.path}`;
      }

      if (mode === 'files_with_matches') {
        const names = fileHits.slice(0, limit).map((hit) => hit.file);
        const extra =
          fileHits.length > names.length ? `\n… ${fileHits.length - names.length} more` : '';
        return `${names.join('\n')}${extra}`;
      }

      if (mode === 'count') {
        const rows = fileHits.slice(0, limit).map((hit) => `${hit.file}:${hit.count}`);
        const extra =
          fileHits.length > rows.length ? `\n… ${fileHits.length - rows.length} more` : '';
        return `${rows.join('\n')}${extra}`;
      }

      const allLines = fileHits.flatMap((hit) => hit.lines);
      const clipped = allLines.slice(0, limit);
      const extra =
        allLines.length > clipped.length ? `\n… ${allLines.length - clipped.length} more` : '';
      return `${clipped.join('\n')}${extra}`;
    },
  });
}
