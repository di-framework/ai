import { systemMessage, userMessage } from '../messages/factories.ts';
import {
  type ChatMessage,
  isSystemMessage,
  isToolResponseMessage,
  isUserMessage,
  type Message,
  type SystemMessage,
  type UserMessage,
} from '../messages/message.ts';
import type { ChatOptions } from './chat-options.ts';
import { mergeChatOptions } from './chat-options.ts';

/**
 * Model request payload, aligned with Spring AI {@code Prompt}.
 *
 * A prompt is a list of {@link Message}s plus optional {@link ChatOptions}.
 */
export class Prompt {
  readonly messages: readonly ChatMessage[];
  readonly options: ChatOptions | undefined;

  constructor(messages: readonly Message[] | string, options?: ChatOptions) {
    if (typeof messages === 'string') {
      this.messages = [userMessage(messages)];
    } else {
      this.messages = messages as readonly ChatMessage[];
    }
    this.options = options;
  }

  /** Spring AI name for the instruction list. */
  get instructions(): readonly ChatMessage[] {
    return this.messages;
  }

  getContents(): string {
    return this.messages.map((m) => m.text ?? '').join('');
  }

  getSystemMessage(): SystemMessage {
    for (const message of this.messages) {
      if (isSystemMessage(message)) return message;
    }
    return systemMessage('');
  }

  getUserMessage(): UserMessage {
    for (let i = this.messages.length - 1; i >= 0; i--) {
      const message = this.messages[i]!;
      if (isUserMessage(message)) return message;
    }
    return userMessage('');
  }

  /**
   * Last user or tool-response message in the prompt.
   * Spring AI: {@code Prompt.getLastUserOrToolResponseMessage}.
   * Returns an empty user message when none is present.
   */
  getLastUserOrToolResponseMessage(): ChatMessage {
    for (let i = this.messages.length - 1; i >= 0; i--) {
      const message = this.messages[i]!;
      if (isUserMessage(message) || isToolResponseMessage(message)) {
        return message;
      }
    }
    return userMessage('');
  }

  getSystemMessages(): readonly SystemMessage[] {
    return this.messages.filter(isSystemMessage);
  }

  getUserMessages(): readonly UserMessage[] {
    return this.messages.filter(isUserMessage);
  }

  copy(): Prompt {
    return new Prompt([...this.messages], this.options ? { ...this.options } : undefined);
  }

  withOptions(options: ChatOptions | undefined): Prompt {
    return new Prompt(this.messages, mergeChatOptions(this.options, options));
  }

  augmentSystemMessage(text: string): Prompt {
    const copy = [...this.messages];
    const index = copy.findIndex(isSystemMessage);
    if (index >= 0) {
      copy[index] = systemMessage(text, copy[index]?.metadata);
    } else {
      copy.unshift(systemMessage(text));
    }
    return new Prompt(copy, this.options);
  }

  augmentUserMessage(text: string): Prompt {
    const copy = [...this.messages];
    for (let i = copy.length - 1; i >= 0; i--) {
      if (isUserMessage(copy[i]!)) {
        const existing = copy[i] as UserMessage;
        copy[i] = userMessage(text, {
          media: existing.media,
          metadata: existing.metadata,
        });
        return new Prompt(copy, this.options);
      }
    }
    copy.push(userMessage(text));
    return new Prompt(copy, this.options);
  }

  static of(text: string, options?: ChatOptions): Prompt {
    return new Prompt(text, options);
  }

  static fromMessages(messages: readonly Message[], options?: ChatOptions): Prompt {
    return new Prompt(messages, options);
  }
}
