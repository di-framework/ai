import { readdirSync } from 'node:fs';
import { join } from 'node:path';
import {
  type AiIgnoreDiscoverySurface,
  type AiIgnorePolicy,
  type AiIgnoreSuppressionDiagnostic,
  aiIgnoreSuppressionDiagnostic,
  evaluateAiIgnorePath,
} from '../policy/index.ts';

export const SKIP_DIR_NAMES = new Set(['node_modules', '.git', 'dist', 'coverage']);

export interface WalkFilesOptions {
  readonly aiIgnorePolicy?: AiIgnorePolicy;
  readonly surface?: AiIgnoreDiscoverySurface;
  readonly onSuppressed?: (diagnostic: AiIgnoreSuppressionDiagnostic) => void;
}

export function walkFiles(
  dir: string,
  depth: number,
  maxDepth: number,
  visit: (file: string) => void,
  options: WalkFilesOptions = {},
): void {
  if (depth > maxDepth) return;
  if (isSuppressed(dir, 'directory', options)) return;
  try {
    const entries = readdirSync(dir, { withFileTypes: true }).sort((left, right) =>
      left.name.localeCompare(right.name),
    );
    for (const entry of entries) {
      if (SKIP_DIR_NAMES.has(entry.name)) continue;
      const full = join(dir, entry.name);
      if (entry.isDirectory()) {
        walkFiles(full, depth + 1, maxDepth, visit, options);
      } else if (entry.isFile() && !isSuppressed(full, 'file', options)) {
        visit(full);
      }
    }
  } catch {
    // Unreadable directories are skipped.
  }
}

function isSuppressed(
  path: string,
  kind: 'file' | 'directory',
  options: WalkFilesOptions,
): boolean {
  if (options.aiIgnorePolicy == null) return false;
  const evaluation = evaluateAiIgnorePath(options.aiIgnorePolicy, path, { kind });
  if (!evaluation.ignored) return false;
  options.onSuppressed?.(
    aiIgnoreSuppressionDiagnostic(evaluation, options.surface ?? 'recursive-walk', kind),
  );
  return true;
}
