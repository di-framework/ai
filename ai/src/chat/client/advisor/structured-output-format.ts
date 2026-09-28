import { mergeChatOptions } from '../../prompt/chat-options.ts';
import { Prompt } from '../../prompt/prompt.ts';
import { ChatClientAttributes } from '../chat-client-attributes.ts';
import { type ChatClientRequest, copyChatClientRequest } from '../chat-client-request.ts';

/**
 * Apply structured-output context to the request before the model call.
 * Spring AI: format / schema injection inside {@code ChatModelCallAdvisor}.
 *
 * - Native mode + schema → set {@code ChatOptions.outputSchema}
 * - Otherwise → append format instructions to the last user message
 */
export function augmentWithFormatInstructions(request: ChatClientRequest): ChatClientRequest {
  const outputFormat = request.context.get(ChatClientAttributes.OUTPUT_FORMAT) as
    | string
    | undefined;
  const outputSchema = request.context.get(ChatClientAttributes.STRUCTURED_OUTPUT_SCHEMA) as
    | string
    | undefined;
  const usesNative = request.context.has(ChatClientAttributes.STRUCTURED_OUTPUT_NATIVE);

  if (!outputFormat && !outputSchema) {
    return request;
  }

  if (usesNative && outputSchema) {
    const options = mergeChatOptions(request.prompt.options, {
      outputSchema,
    });
    return copyChatClientRequest(request, {
      prompt: new Prompt(request.prompt.messages, options),
    });
  }

  if (!outputFormat) {
    return request;
  }

  const currentUser = request.prompt.getUserMessage().text ?? '';
  const augmented = currentUser ? `${currentUser}\n${outputFormat}` : outputFormat;
  return copyChatClientRequest(request, {
    prompt: request.prompt.augmentUserMessage(augmented),
  });
}
