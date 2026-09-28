/**
 * Node {@code errno} helpers so callers can act on I/O failures instead of
 * exists/stat-then-open races.
 */
export function nodeErrnoCode(error: unknown): string | undefined {
  if (error == null || typeof error !== 'object' || !('code' in error)) {
    return undefined;
  }
  const code = (error as { code: unknown }).code;
  return typeof code === 'string' ? code : undefined;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
