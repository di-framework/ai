export type SkillsIndexOperation = 'build' | 'inspect' | 'validate' | 'query' | 'migrate';

export type SkillsIndexOperationErrorCode =
  | 'INVALID_OPTIONS'
  | 'SOURCE_NOT_FOUND'
  | 'INDEX_NOT_FOUND'
  | 'INVALID_INDEX'
  | 'SOURCE_DRIFT'
  | 'EMBEDDING_FAILED'
  | 'WRITE_FAILED'
  | 'OPERATION_FAILED';

/** Stable failure boundary shared by every programmatic skills-index operation. */
export class SkillsIndexOperationError extends Error {
  override readonly name = 'SkillsIndexOperationError';

  constructor(
    readonly operation: SkillsIndexOperation,
    readonly code: SkillsIndexOperationErrorCode,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
  }
}

export function asSkillsIndexOperationError(
  operation: SkillsIndexOperation,
  code: SkillsIndexOperationErrorCode,
  error: unknown,
): SkillsIndexOperationError {
  if (error instanceof SkillsIndexOperationError) return error;
  const message = error instanceof Error ? error.message : String(error);
  return new SkillsIndexOperationError(operation, code, message, { cause: error });
}
