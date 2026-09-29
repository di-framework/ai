/**
 * Normalized AI error codes for provider and framework failures.
 */
export type AiErrorCode =
  | 'authentication'
  | 'authorization'
  | 'rate-limit'
  | 'invalid-request'
  | 'model-unavailable'
  | 'timeout'
  | 'cancelled'
  | 'provider-error'
  | 'tool-validation'
  | 'tool-execution'
  | 'output-validation';

export interface AiErrorDetails {
  readonly provider?: string;
  readonly model?: string;
  readonly requestId?: string;
  readonly retryable?: boolean;
  readonly status?: number;
  readonly cause?: unknown;
}

export class AiError extends Error {
  readonly code: AiErrorCode;
  readonly details: AiErrorDetails;

  constructor(message: string, code: AiErrorCode, details: AiErrorDetails = {}) {
    super(message);
    this.name = 'AiError';
    this.code = code;
    this.details = details;
  }
}

export function isAiError(error: unknown): error is AiError {
  return error instanceof AiError;
}

export function cancelledError(
  message = 'Request was cancelled',
  details?: AiErrorDetails,
): AiError {
  return new AiError(message, 'cancelled', { retryable: false, ...details });
}
