import { CHAT_MEMORY_CONVERSATION_ID, type ChatMemory, messagesEqual } from '../../memory/index.ts';
import { type ChatMessage, isSystemMessage, type Message } from '../../messages/message.ts';
import { Prompt } from '../../prompt/prompt.ts';
import { type ChatClientRequest, copyChatClientRequest } from '../chat-client-request.ts';
import type { ChatClientResponse } from '../chat-client-response.ts';
import type {
  CallAdvisor,
  CallAdvisorChain,
  StreamAdvisor,
  StreamAdvisorChain,
} from './advisor.ts';
import { DEFAULT_CHAT_MEMORY_PRECEDENCE_ORDER } from './ordered.ts';

export interface MessageChatMemoryAdvisorOptions {
  readonly chatMemory: ChatMemory;
  /** Default {@link DEFAULT_CHAT_MEMORY_PRECEDENCE_ORDER}. */
  readonly order?: number;
}

/**
 * Advisor that loads conversation history into the prompt before the model
 * call and appends user/assistant messages afterward.
 * Spring AI: {@code MessageChatMemoryAdvisor}.
 *
 * Order defaults to {@link DEFAULT_CHAT_MEMORY_PRECEDENCE_ORDER} so it wraps
 * {@code ToolCallingAdvisor} and does not re-run on every tool iteration.
 *
 * Requires {@link CHAT_MEMORY_CONVERSATION_ID} in the request context
 * (via {@code advisorContext} / default context).
 */
export class MessageChatMemoryAdvisor implements CallAdvisor, StreamAdvisor {
  readonly name = 'Message Chat Memory Advisor';
  readonly order: number;
  private readonly chatMemory: ChatMemory;

  constructor(options: MessageChatMemoryAdvisorOptions) {
    if (options.chatMemory == null) {
      throw new Error('chatMemory cannot be null');
    }
    this.chatMemory = options.chatMemory;
    this.order = options.order ?? DEFAULT_CHAT_MEMORY_PRECEDENCE_ORDER;
  }

  static builder(chatMemory: ChatMemory): MessageChatMemoryAdvisorBuilder {
    return new MessageChatMemoryAdvisorBuilder(chatMemory);
  }

  static of(chatMemory: ChatMemory, order?: number): MessageChatMemoryAdvisor {
    return new MessageChatMemoryAdvisor({ chatMemory, order });
  }

  /**
   * Resolve conversation id from advisor context.
   * Spring AI: {@code BaseChatMemoryAdvisor.getConversationId}.
   */
  getConversationId(context: Map<string, unknown> | ReadonlyMap<string, unknown>): string {
    const value = context.get(CHAT_MEMORY_CONVERSATION_ID);
    if (value == null) {
      throw new Error(
        `conversationId cannot be null — set context key "${CHAT_MEMORY_CONVERSATION_ID}"`,
      );
    }
    return String(value);
  }

  before(
    request: ChatClientRequest,
    _chain?: CallAdvisorChain | StreamAdvisorChain,
  ): ChatClientRequest {
    const conversationId = this.getConversationId(request.context);
    const memoryMessages = this.chatMemory.get(conversationId);
    const promptMessages = request.prompt.messages;

    const processedMessages: ChatMessage[] = [];
    if (!isMemoryAlreadyInPrompt(promptMessages, memoryMessages)) {
      processedMessages.push(...(memoryMessages as ChatMessage[]));
    }
    processedMessages.push(...promptMessages);

    // Ensure the first system message, if any, appears first.
    for (let i = 0; i < processedMessages.length; i++) {
      if (isSystemMessage(processedMessages[i]!)) {
        const [system] = processedMessages.splice(i, 1);
        processedMessages.unshift(system!);
        break;
      }
    }

    const processed = copyChatClientRequest(request, {
      prompt: new Prompt(processedMessages, request.prompt.options),
    });

    // Persist the new user / tool-response turn message.
    const toStore = processed.prompt.getLastUserOrToolResponseMessage();
    this.chatMemory.add(conversationId, [toStore]);

    return processed;
  }

  after(
    response: ChatClientResponse,
    _chain?: CallAdvisorChain | StreamAdvisorChain,
  ): ChatClientResponse {
    const conversationId = this.getConversationId(response.context);
    const assistantMessages: ChatMessage[] = [];
    if (response.chatResponse) {
      for (const generation of response.chatResponse.results) {
        assistantMessages.push(generation.output);
      }
    }
    if (assistantMessages.length > 0) {
      this.chatMemory.add(conversationId, assistantMessages);
    }
    return response;
  }

  async adviseCall(
    request: ChatClientRequest,
    chain: CallAdvisorChain,
  ): Promise<ChatClientResponse> {
    const processed = this.before(request, chain);
    const response = await chain.nextCall(processed);
    return this.after(response, chain);
  }

  async *adviseStream(
    request: ChatClientRequest,
    chain: StreamAdvisorChain,
  ): AsyncIterable<ChatClientResponse> {
    const processed = this.before(request, chain);
    let last: ChatClientResponse | undefined;
    for await (const response of chain.nextStream(processed)) {
      last = response;
      yield response;
    }
    // Persist assistant once the stream completes (full last chunk / final response).
    if (last) {
      this.after(last, chain);
    }
  }
}

export class MessageChatMemoryAdvisorBuilder {
  private orderValue: number | undefined;

  constructor(private readonly chatMemory: ChatMemory) {
    if (chatMemory == null) {
      throw new Error('chatMemory cannot be null');
    }
  }

  order(order: number): this {
    this.orderValue = order;
    return this;
  }

  build(): MessageChatMemoryAdvisor {
    return new MessageChatMemoryAdvisor({
      chatMemory: this.chatMemory,
      order: this.orderValue,
    });
  }
}

function isMemoryAlreadyInPrompt(
  promptMessages: readonly ChatMessage[],
  memoryMessages: readonly Message[],
): boolean {
  if (memoryMessages.length === 0) {
    return true;
  }
  if (promptMessages.length < memoryMessages.length) {
    return false;
  }
  for (let offset = 0; offset <= promptMessages.length - memoryMessages.length; offset++) {
    if (startsWith(promptMessages, memoryMessages, offset)) {
      return true;
    }
  }
  return false;
}

function startsWith(
  messages: readonly Message[],
  prefix: readonly Message[],
  offset: number,
): boolean {
  if (messages.length - offset < prefix.length) {
    return false;
  }
  for (let i = 0; i < prefix.length; i++) {
    if (!messagesEqual(messages[i + offset]!, prefix[i]!)) {
      return false;
    }
  }
  return true;
}
