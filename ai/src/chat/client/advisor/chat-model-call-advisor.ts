import type { ToolCallingManager } from '../../../model/tool/tool-calling-manager.ts';
import type { ChatModel } from '../../model/chat-model.ts';
import {
  CALL_WITH_TOOL_MANAGER,
  isToolExecutingChatModel,
  TOOL_MANAGER_CONTEXT,
} from '../../model/tool-execution-context.ts';
import type { ChatClientRequest } from '../chat-client-request.ts';
import { chatClientResponse } from '../chat-client-response.ts';
import type { CallAdvisor, CallAdvisorChain } from './advisor.ts';
import { LOWEST_PRECEDENCE } from './ordered.ts';
import { augmentWithFormatInstructions } from './structured-output-format.ts';

/**
 * Terminal call advisor that invokes {@link ChatModel.call}.
 * Spring AI: {@code ChatModelCallAdvisor}.
 */
export class ChatModelCallAdvisor implements CallAdvisor {
  readonly name = 'ChatModelCallAdvisor';
  readonly order = LOWEST_PRECEDENCE;

  constructor(private readonly chatModel: ChatModel) {}

  async adviseCall(
    request: ChatClientRequest,
    _chain: CallAdvisorChain,
  ): Promise<ReturnType<typeof chatClientResponse>> {
    const formatted = augmentWithFormatInstructions(request);
    const chatResponse = isToolExecutingChatModel(this.chatModel)
      ? await this.chatModel[CALL_WITH_TOOL_MANAGER](
          formatted.prompt,
          formatted.context.get(TOOL_MANAGER_CONTEXT) as ToolCallingManager | undefined,
        )
      : await this.chatModel.call(formatted.prompt);
    return chatClientResponse(chatResponse, formatted.context);
  }

  static of(chatModel: ChatModel): ChatModelCallAdvisor {
    return new ChatModelCallAdvisor(chatModel);
  }
}
