import {
  type SchemaValidationResult,
  validateAgainstJsonSchema,
} from '../../../converter/json-schema-validator.ts';
import {
  defaultResponseTextCleaner,
  type ResponseTextCleaner,
} from '../../../converter/response-text-cleaner.ts';
import { Prompt } from '../../prompt/prompt.ts';
import { type ChatClientRequest, copyChatClientRequest } from '../chat-client-request.ts';
import type { ChatClientResponse } from '../chat-client-response.ts';
import type {
  CallAdvisor,
  CallAdvisorChain,
  StreamAdvisor,
  StreamAdvisorChain,
} from './advisor.ts';
import { LOWEST_PRECEDENCE } from './ordered.ts';

export interface StructuredOutputValidationAdvisorOptions {
  readonly outputJsonSchema: string;
  /** Extra retries after the first attempt. Default 3 (Spring AI). */
  readonly maxRepeatAttempts?: number;
  /** Default {@code LOWEST_PRECEDENCE - 2000}. */
  readonly order?: number;
  readonly textCleaner?: ResponseTextCleaner;
}

/**
 * Validates structured JSON against a schema and re-prompts on failure.
 * Spring AI: {@code StructuredOutputValidationAdvisor}.
 *
 * Streaming is not validated (passes through); use call path for retries.
 */
export class StructuredOutputValidationAdvisor implements CallAdvisor, StreamAdvisor {
  readonly name = 'Structured Output Validation Advisor';
  readonly order: number;
  private readonly outputJsonSchema: string;
  private readonly maxRepeatAttempts: number;
  private readonly textCleaner: ResponseTextCleaner;

  constructor(options: StructuredOutputValidationAdvisorOptions) {
    if (!options.outputJsonSchema?.trim()) {
      throw new Error('outputJsonSchema must not be empty');
    }
    this.outputJsonSchema = options.outputJsonSchema;
    this.maxRepeatAttempts = options.maxRepeatAttempts ?? 3;
    if (this.maxRepeatAttempts < 0) {
      throw new Error('maxRepeatAttempts must be >= 0');
    }
    this.order = options.order ?? LOWEST_PRECEDENCE - 2000;
    this.textCleaner = options.textCleaner ?? defaultResponseTextCleaner;
  }

  static builder(
    options: StructuredOutputValidationAdvisorOptions,
  ): StructuredOutputValidationAdvisor {
    return new StructuredOutputValidationAdvisor(options);
  }

  async adviseCall(
    request: ChatClientRequest,
    chain: CallAdvisorChain,
  ): Promise<ChatClientResponse> {
    let processed = request;
    let lastResponse: ChatClientResponse | undefined;
    let success = false;

    // attempts = 1 + maxRepeatAttempts (Spring AI semantics)
    for (let remaining = 1 + this.maxRepeatAttempts; remaining > 0 && !success; remaining--) {
      lastResponse =
        typeof chain.copy === 'function'
          ? await chain.copy(this).nextCall(processed)
          : await chain.nextCall(processed);

      const chatResponse = lastResponse.chatResponse;
      // Skip validation for tool-call rounds
      if (chatResponse?.hasToolCalls()) {
        success = true;
        break;
      }

      const validation = this.validateResponse(lastResponse);
      success = validation.success;
      if (!success && remaining > 1) {
        const errorMessage = `Output JSON validation failed because of: ${validation.errorMessage}`;
        const userText = request.prompt.getUserMessage().text ?? '';
        const augmented = userText ? `${userText}\n${errorMessage}` : errorMessage;
        processed = copyChatClientRequest(request, {
          prompt: new Prompt(
            request.prompt.augmentUserMessage(augmented).messages,
            request.prompt.options,
          ),
        });
      }
    }

    return lastResponse!;
  }

  async *adviseStream(
    request: ChatClientRequest,
    chain: StreamAdvisorChain,
  ): AsyncIterable<ChatClientResponse> {
    // Streaming validation not supported (matches Spring AI).
    yield* chain.nextStream(request);
  }

  private validateResponse(response: ChatClientResponse): SchemaValidationResult {
    const text = response.chatResponse?.getResult()?.output.text;
    if (text == null) {
      return {
        success: false,
        errorMessage: 'Missing required json output for validation.',
      };
    }
    const cleaned = this.textCleaner(text);
    if (!cleaned.trim()) {
      return {
        success: false,
        errorMessage: 'Empty JSON output for validation.',
      };
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(cleaned);
    } catch (cause) {
      return {
        success: false,
        errorMessage: `Invalid JSON: ${cause instanceof Error ? cause.message : String(cause)}`,
      };
    }
    return validateAgainstJsonSchema(parsed, this.outputJsonSchema);
  }
}
