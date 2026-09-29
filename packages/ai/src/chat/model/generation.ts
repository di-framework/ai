import { assistantMessage } from '../messages/factories.ts';
import type { AssistantMessage } from '../messages/message.ts';

/**
 * One generation result, aligned with Spring AI {@code Generation}.
 */
export interface GenerationMetadata {
  readonly finishReason?: string;
  readonly [key: string]: unknown;
}

export interface Generation {
  readonly output: AssistantMessage;
  readonly metadata: GenerationMetadata;
}

export function generation(
  output: AssistantMessage | string,
  metadata: GenerationMetadata = {},
): Generation {
  return {
    output: typeof output === 'string' ? assistantMessage(output) : output,
    metadata,
  };
}
