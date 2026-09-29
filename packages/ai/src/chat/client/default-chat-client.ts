import type { Media } from '../../content/media.ts';
import {
  isStructuredOutputConverter,
  type StructuredOutputConverter,
  schemaOutputConverter,
} from '../../converter/index.ts';
import { toolCallbacksFromBean } from '../../di/tool-decorator.ts';
import type { ToolCallback } from '../../tool/tool-callback.ts';
import type { ToolCallbackProvider } from '../../tool/tool-callback-provider.ts';
import {
  resolveToolCallbacks,
  resolveToolCallbacksDedupe,
} from '../../tool/tool-callback-provider.ts';
import { systemMessage, userMessage } from '../messages/factories.ts';
import type { ChatMessage, Message } from '../messages/message.ts';
import type { ChatModel } from '../model/chat-model.ts';
import type { ChatResponse } from '../model/chat-response.ts';
import type { ChatOptions } from '../prompt/chat-options.ts';
import { mergeChatOptions } from '../prompt/chat-options.ts';
import { Prompt } from '../prompt/prompt.ts';
import {
  type Advisor,
  ChatModelCallAdvisor,
  ChatModelStreamAdvisor,
  DefaultAdvisorChain,
  isToolAdvisor,
  StructuredOutputValidationAdvisor,
  ToolCallingAdvisor,
  type ToolCallingAdvisorOptions,
} from './advisor/index.ts';
import { ChatClientAttributes } from './chat-client-attributes.ts';
import { type ChatClientRequest, chatClientRequest } from './chat-client-request.ts';
import type { ChatClientResponse } from './chat-client-response.ts';
import { renderTemplate } from './template.ts';

/** Context / builder flag to skip auto-registration of ToolCallingAdvisor. */
export const TOOL_CALLING_ADVISOR_AUTO_REGISTER = 'toolCallingAdvisorAutoRegister';

export type ToolSource = ToolCallback | ToolCallbackProvider | readonly ToolCallback[];

export interface ChatClientBuilderOptions {
  readonly defaultSystem?: string;
  readonly defaultUser?: string;
  readonly defaultOptions?: ChatOptions;
  readonly defaultAdvisors?: readonly Advisor[];
  readonly defaultContext?: Readonly<Record<string, unknown>>;
  readonly defaultToolCallbacks?: readonly ToolCallback[];
  readonly defaultToolContext?: Readonly<Record<string, unknown>>;
  /**
   * Options used when auto-registering {@link ToolCallingAdvisor}.
   * Ignored when a ToolAdvisor is already present in the chain.
   */
  readonly toolCallingAdvisorOptions?: ToolCallingAdvisorOptions;
  /**
   * When false, do not auto-register {@link ToolCallingAdvisor}.
   * Default true (Spring AI 2.x behavior).
   */
  readonly autoRegisterToolCallingAdvisor?: boolean;
}

export interface ChatClient {
  prompt(): ChatClientRequestSpec;
  prompt(content: string): ChatClientRequestSpec;
  prompt(prompt: Prompt): ChatClientRequestSpec;
  mutate(): ChatClientBuilder;
}

export interface ChatClientBuilder {
  defaultSystem(text: string): ChatClientBuilder;
  defaultUser(text: string): ChatClientBuilder;
  defaultOptions(options: ChatOptions): ChatClientBuilder;
  defaultAdvisors(...advisors: Advisor[]): ChatClientBuilder;
  defaultContext(context: Readonly<Record<string, unknown>>): ChatClientBuilder;
  defaultTools(...tools: ToolSource[]): ChatClientBuilder;
  /** Expand `@Tool` methods on bean instances into default tools. */
  defaultToolBeans(...beans: readonly object[]): ChatClientBuilder;
  defaultToolContext(context: Readonly<Record<string, unknown>>): ChatClientBuilder;
  toolCallingAdvisorOptions(options: ToolCallingAdvisorOptions): ChatClientBuilder;
  autoRegisterToolCallingAdvisor(enabled: boolean): ChatClientBuilder;
  build(): ChatClient;
}

/**
 * Options for {@link CallResponseSpec.entity} / {@code responseEntity}.
 * Spring AI: {@code ChatClient.EntityParamSpec}.
 */
export interface EntityParamSpec {
  /**
   * Use provider-native structured output (sets {@code ChatOptions.outputSchema})
   * instead of appending format instructions to the user message.
   */
  useProviderStructuredOutput(): EntityParamSpec;
  /**
   * Validate output against the converter JSON schema and re-prompt on failure.
   */
  validateSchema(): EntityParamSpec;
}

/**
 * Pair of raw {@link ChatResponse} and converted entity.
 * Spring AI: {@code ResponseEntity<ChatResponse, T>}.
 */
export interface EntityResponse<T> {
  readonly chatResponse: ChatResponse | undefined;
  readonly entity: T | undefined;
}

export type EntityInput<T> = StructuredOutputConverter<T> | string | Record<string, unknown>;

export interface CallResponseSpec {
  content(): Promise<string | undefined>;
  chatResponse(): Promise<ChatResponse | undefined>;
  chatClientResponse(): Promise<ChatClientResponse>;
  /**
   * Convert the model reply with a structured-output converter or JSON schema.
   * Spring AI: {@code CallResponseSpec.entity}.
   */
  entity<T>(
    converterOrSchema: EntityInput<T>,
    configure?: (spec: EntityParamSpec) => void,
  ): Promise<T | undefined>;
  /**
   * Like {@link entity} but also returns the raw {@link ChatResponse}.
   * Spring AI: {@code CallResponseSpec.responseEntity}.
   */
  responseEntity<T>(
    converterOrSchema: EntityInput<T>,
    configure?: (spec: EntityParamSpec) => void,
  ): Promise<EntityResponse<T>>;
}

export interface StreamResponseSpec {
  content(): AsyncIterable<string>;
  chatResponse(): AsyncIterable<ChatResponse>;
  chatClientResponse(): AsyncIterable<ChatClientResponse>;
}

export interface ChatClientRequestSpec {
  system(text: string): ChatClientRequestSpec;
  system(
    spec: (s: { text: string; params?: Record<string, unknown> }) => void,
  ): ChatClientRequestSpec;
  user(text: string): ChatClientRequestSpec;
  user(
    spec: (u: { text: string; params?: Record<string, unknown>; media?: readonly Media[] }) => void,
  ): ChatClientRequestSpec;
  messages(...messages: Message[]): ChatClientRequestSpec;
  messages(messages: readonly Message[]): ChatClientRequestSpec;
  options(options: ChatOptions): ChatClientRequestSpec;
  advisors(...advisors: Advisor[]): ChatClientRequestSpec;
  advisors(advisors: readonly Advisor[]): ChatClientRequestSpec;
  /** Merge values into the advisor context map. */
  advisorContext(context: Readonly<Record<string, unknown>>): ChatClientRequestSpec;
  /**
   * Register tools for this call (replaces any previously set call-level tools;
   * merges with builder defaults when call-level tools are empty).
   * Spring AI: {@code ChatClientRequestSpec.tools}.
   */
  tools(...tools: ToolSource[]): ChatClientRequestSpec;
  /** Merge values into the tool context passed to callbacks. */
  toolContext(context: Readonly<Record<string, unknown>>): ChatClientRequestSpec;
  call(): CallResponseSpec;
  stream(): StreamResponseSpec;
  /** Build the immutable request without executing (useful for tests). */
  toRequest(): ChatClientRequest;
}

class DefaultEntityParamSpec implements EntityParamSpec {
  enableNative = false;
  validated = false;

  useProviderStructuredOutput(): EntityParamSpec {
    this.enableNative = true;
    return this;
  }

  validateSchema(): EntityParamSpec {
    this.validated = true;
    return this;
  }
}

class DefaultCallResponseSpec implements CallResponseSpec {
  constructor(
    private readonly request: ChatClientRequest,
    /** Advisors excluding the terminal model advisor (includes auto tool advisor). */
    private readonly advisors: readonly Advisor[],
    private readonly chatModel: ChatModel,
  ) {}

  private buildChain(extra: readonly Advisor[] = []): DefaultAdvisorChain {
    return DefaultAdvisorChain.of([
      ...this.advisors,
      ...extra,
      ChatModelCallAdvisor.of(this.chatModel),
    ]);
  }

  async chatClientResponse(): Promise<ChatClientResponse> {
    return this.buildChain().nextCall(this.request);
  }

  async chatResponse(): Promise<ChatResponse | undefined> {
    return (await this.chatClientResponse()).chatResponse;
  }

  async content(): Promise<string | undefined> {
    const response = await this.chatResponse();
    if (!response?.getResult()) return undefined;
    return response.content;
  }

  async entity<T>(
    converterOrSchema: EntityInput<T>,
    configure?: (spec: EntityParamSpec) => void,
  ): Promise<T | undefined> {
    const result = await this.responseEntity(converterOrSchema, configure);
    return result.entity;
  }

  async responseEntity<T>(
    converterOrSchema: EntityInput<T>,
    configure?: (spec: EntityParamSpec) => void,
  ): Promise<EntityResponse<T>> {
    const converter = resolveEntityConverter<T>(converterOrSchema);
    const params = new DefaultEntityParamSpec();
    configure?.(params);

    // Context is mutable for this request (Spring AI puts format keys here).
    const context = this.request.context;
    const format = converter.getFormat();
    if (format) {
      context.set(ChatClientAttributes.OUTPUT_FORMAT, format);
    }

    if (params.enableNative) {
      context.set(ChatClientAttributes.STRUCTURED_OUTPUT_NATIVE, true);
      const schema = converter.getJsonSchema();
      if (schema) {
        context.set(ChatClientAttributes.STRUCTURED_OUTPUT_SCHEMA, schema);
      }
    }

    const extra: Advisor[] = [];
    if (params.validated) {
      const schema = converter.getJsonSchema();
      if (!schema) {
        throw new Error('validateSchema() requires a converter with a JSON schema');
      }
      extra.push(
        new StructuredOutputValidationAdvisor({
          outputJsonSchema: schema,
        }),
      );
    }

    const chain = this.buildChain(extra);
    const clientResponse = await chain.nextCall(this.request);
    const chatResponse = clientResponse.chatResponse;
    const text = chatResponse?.getResult()?.output.text;
    if (text == null || text === '') {
      return { chatResponse, entity: undefined };
    }
    return {
      chatResponse,
      entity: converter.convert(text),
    };
  }
}

function resolveEntityConverter<T>(
  converterOrSchema: EntityInput<T>,
): StructuredOutputConverter<T> {
  if (isStructuredOutputConverter(converterOrSchema)) {
    return converterOrSchema as StructuredOutputConverter<T>;
  }
  return schemaOutputConverter<T>({
    schema: converterOrSchema as string | Record<string, unknown>,
  });
}

class DefaultStreamResponseSpec implements StreamResponseSpec {
  constructor(
    private readonly request: ChatClientRequest,
    private readonly chain: DefaultAdvisorChain,
  ) {}

  async *chatClientResponse(): AsyncIterable<ChatClientResponse> {
    yield* this.chain.nextStream(this.request);
  }

  async *chatResponse(): AsyncIterable<ChatResponse> {
    for await (const clientResponse of this.chatClientResponse()) {
      if (clientResponse.chatResponse) {
        yield clientResponse.chatResponse;
      }
    }
  }

  async *content(): AsyncIterable<string> {
    for await (const response of this.chatResponse()) {
      const text = response.content;
      if (text) yield text;
    }
  }
}

class DefaultChatClientRequestSpec implements ChatClientRequestSpec {
  private systemText: string | undefined;
  private systemParams: Record<string, unknown> = {};
  private userText: string | undefined;
  private userParams: Record<string, unknown> = {};
  private userMedia: readonly Media[] = [];
  private readonly messageList: ChatMessage[] = [];
  private optionsValue: ChatOptions | undefined;
  private readonly advisorList: Advisor[] = [];
  private readonly context: Map<string, unknown> = new Map();
  private callToolCallbacks: ToolCallback[] | undefined;
  private toolContextValue: Record<string, unknown> = {};

  constructor(
    private readonly chatModel: ChatModel,
    private readonly defaults: ChatClientBuilderOptions,
  ) {
    this.systemText = defaults.defaultSystem;
    this.userText = defaults.defaultUser;
    this.optionsValue = defaults.defaultOptions
      ? { ...defaults.defaultOptions }
      : chatModel.options
        ? { ...chatModel.options }
        : undefined;
    if (defaults.defaultAdvisors) {
      this.advisorList.push(...defaults.defaultAdvisors);
    }
    if (defaults.defaultContext) {
      for (const [k, v] of Object.entries(defaults.defaultContext)) {
        this.context.set(k, v);
      }
    }
    if (defaults.defaultToolContext) {
      Object.assign(this.toolContextValue, defaults.defaultToolContext);
    }
  }

  system(
    textOrSpec: string | ((s: { text: string; params?: Record<string, unknown> }) => void),
  ): ChatClientRequestSpec {
    if (typeof textOrSpec === 'string') {
      this.systemText = textOrSpec;
      return this;
    }
    const bag: { text: string; params?: Record<string, unknown> } = {
      text: this.systemText ?? '',
    };
    textOrSpec(bag);
    this.systemText = bag.text;
    if (bag.params) Object.assign(this.systemParams, bag.params);
    return this;
  }

  user(
    textOrSpec:
      | string
      | ((u: { text: string; params?: Record<string, unknown>; media?: readonly Media[] }) => void),
  ): ChatClientRequestSpec {
    if (typeof textOrSpec === 'string') {
      this.userText = textOrSpec;
      return this;
    }
    const bag: {
      text: string;
      params?: Record<string, unknown>;
      media?: readonly Media[];
    } = {
      text: this.userText ?? '',
    };
    textOrSpec(bag);
    this.userText = bag.text;
    if (bag.params) Object.assign(this.userParams, bag.params);
    if (bag.media) this.userMedia = bag.media;
    return this;
  }

  messages(...args: Array<Message | readonly Message[]>): ChatClientRequestSpec {
    for (const arg of args) {
      if (Array.isArray(arg)) {
        this.messageList.push(...(arg as ChatMessage[]));
      } else {
        this.messageList.push(arg as ChatMessage);
      }
    }
    return this;
  }

  options(options: ChatOptions): ChatClientRequestSpec {
    this.optionsValue = mergeChatOptions(this.optionsValue, options);
    return this;
  }

  advisors(...args: Array<Advisor | readonly Advisor[]>): ChatClientRequestSpec {
    for (const arg of args) {
      if (Array.isArray(arg)) {
        this.advisorList.push(...arg);
      } else {
        this.advisorList.push(arg as Advisor);
      }
    }
    return this;
  }

  advisorContext(context: Readonly<Record<string, unknown>>): ChatClientRequestSpec {
    for (const [k, v] of Object.entries(context)) {
      this.context.set(k, v);
    }
    return this;
  }

  tools(...tools: ToolSource[]): ChatClientRequestSpec {
    this.callToolCallbacks = resolveToolCallbacks(...tools);
    return this;
  }

  toolContext(context: Readonly<Record<string, unknown>>): ChatClientRequestSpec {
    Object.assign(this.toolContextValue, context);
    return this;
  }

  toRequest(): ChatClientRequest {
    const messages: ChatMessage[] = [];

    if (this.systemText != null && this.systemText !== '') {
      const text =
        Object.keys(this.systemParams).length > 0
          ? renderTemplate(this.systemText, this.systemParams)
          : this.systemText;
      messages.push(systemMessage(text));
    }

    messages.push(...this.messageList);

    if (this.userText != null && this.userText !== '') {
      const text =
        Object.keys(this.userParams).length > 0
          ? renderTemplate(this.userText, this.userParams)
          : this.userText;
      messages.push(
        userMessage(text, {
          media: this.userMedia,
        }),
      );
    }

    const modelDefaults = this.chatModel.options;
    let options = mergeChatOptions(modelDefaults, this.optionsValue);

    const toolCallbacks = this.resolveEffectiveToolCallbacks();
    const toolContext =
      Object.keys(this.toolContextValue).length > 0
        ? { ...this.toolContextValue }
        : options?.toolContext;

    if (toolCallbacks !== undefined || toolContext !== undefined) {
      options = mergeChatOptions(options, {
        toolCallbacks,
        toolContext,
      });
    }

    const prompt = new Prompt(messages, options);
    return chatClientRequest(prompt, this.context);
  }

  private resolveEffectiveToolCallbacks(): readonly ToolCallback[] | undefined {
    if (this.callToolCallbacks !== undefined) {
      return this.callToolCallbacks;
    }
    if (this.defaults.defaultToolCallbacks !== undefined) {
      return this.defaults.defaultToolCallbacks;
    }
    return this.optionsValue?.toolCallbacks ?? this.chatModel.options?.toolCallbacks;
  }

  /** Advisors for the call/stream chain, excluding terminal model advisors. */
  private collectAdvisors(): Advisor[] {
    const advisors: Advisor[] = [...this.advisorList];
    this.autoRegisterToolCallingAdvisor(advisors);
    return advisors;
  }

  private autoRegisterToolCallingAdvisor(advisors: Advisor[]): void {
    const autoRegisterDisabled =
      this.defaults.autoRegisterToolCallingAdvisor === false ||
      this.context.get(TOOL_CALLING_ADVISOR_AUTO_REGISTER) === false;

    if (autoRegisterDisabled) return;

    if (advisors.some(isToolAdvisor)) return;

    advisors.push(ToolCallingAdvisor.of(this.defaults.toolCallingAdvisorOptions));
  }

  call(): CallResponseSpec {
    const request = this.toRequest();
    return new DefaultCallResponseSpec(request, this.collectAdvisors(), this.chatModel);
  }

  stream(): StreamResponseSpec {
    const request = this.toRequest();
    const chain = DefaultAdvisorChain.of([
      ...this.collectAdvisors(),
      ChatModelStreamAdvisor.of(this.chatModel),
    ]);
    return new DefaultStreamResponseSpec(request, chain);
  }
}

class DefaultChatClient implements ChatClient {
  constructor(
    private readonly chatModel: ChatModel,
    private readonly defaults: ChatClientBuilderOptions = {},
  ) {}

  prompt(): ChatClientRequestSpec;
  prompt(content: string): ChatClientRequestSpec;
  prompt(prompt: Prompt): ChatClientRequestSpec;
  prompt(input?: string | Prompt): ChatClientRequestSpec {
    const spec = new DefaultChatClientRequestSpec(this.chatModel, this.defaults);
    if (input === undefined) return spec;
    if (typeof input === 'string') {
      return spec.user(input);
    }
    return spec.messages(input.messages).options(input.options ?? {});
  }

  mutate(): ChatClientBuilder {
    return new DefaultChatClientBuilder(this.chatModel, { ...this.defaults });
  }
}

class DefaultChatClientBuilder implements ChatClientBuilder {
  private options: ChatClientBuilderOptions;

  constructor(
    private readonly chatModel: ChatModel,
    options: ChatClientBuilderOptions = {},
  ) {
    this.options = {
      ...options,
      defaultAdvisors: options.defaultAdvisors ? [...options.defaultAdvisors] : [],
      defaultContext: options.defaultContext ? { ...options.defaultContext } : {},
      defaultOptions: options.defaultOptions ? { ...options.defaultOptions } : undefined,
      defaultToolCallbacks: options.defaultToolCallbacks
        ? [...options.defaultToolCallbacks]
        : undefined,
      defaultToolContext: options.defaultToolContext
        ? { ...options.defaultToolContext }
        : undefined,
    };
  }

  defaultSystem(text: string): ChatClientBuilder {
    this.options = { ...this.options, defaultSystem: text };
    return this;
  }

  defaultUser(text: string): ChatClientBuilder {
    this.options = { ...this.options, defaultUser: text };
    return this;
  }

  defaultOptions(options: ChatOptions): ChatClientBuilder {
    this.options = {
      ...this.options,
      defaultOptions: mergeChatOptions(this.options.defaultOptions, options),
    };
    return this;
  }

  defaultAdvisors(...advisors: Advisor[]): ChatClientBuilder {
    const existing = this.options.defaultAdvisors ?? [];
    this.options = {
      ...this.options,
      defaultAdvisors: [...existing, ...advisors],
    };
    return this;
  }

  defaultContext(context: Readonly<Record<string, unknown>>): ChatClientBuilder {
    this.options = {
      ...this.options,
      defaultContext: { ...this.options.defaultContext, ...context },
    };
    return this;
  }

  defaultTools(...tools: ToolSource[]): ChatClientBuilder {
    this.options = {
      ...this.options,
      defaultToolCallbacks: resolveToolCallbacksDedupe(this.options.defaultToolCallbacks, ...tools),
    };
    return this;
  }

  defaultToolBeans(...beans: readonly object[]): ChatClientBuilder {
    const fromBeans = beans.flatMap((bean) => toolCallbacksFromBean(bean));
    return this.defaultTools(...fromBeans);
  }

  defaultToolContext(context: Readonly<Record<string, unknown>>): ChatClientBuilder {
    this.options = {
      ...this.options,
      defaultToolContext: {
        ...this.options.defaultToolContext,
        ...context,
      },
    };
    return this;
  }

  toolCallingAdvisorOptions(options: ToolCallingAdvisorOptions): ChatClientBuilder {
    this.options = {
      ...this.options,
      toolCallingAdvisorOptions: {
        ...this.options.toolCallingAdvisorOptions,
        ...options,
      },
    };
    return this;
  }

  autoRegisterToolCallingAdvisor(enabled: boolean): ChatClientBuilder {
    this.options = {
      ...this.options,
      autoRegisterToolCallingAdvisor: enabled,
    };
    return this;
  }

  build(): ChatClient {
    return new DefaultChatClient(this.chatModel, this.options);
  }
}

/**
 * Spring AI-style factory helpers for {@link ChatClient}.
 */
export const ChatClient = {
  create(chatModel: ChatModel): ChatClient {
    return new DefaultChatClient(chatModel);
  },

  builder(chatModel: ChatModel): ChatClientBuilder {
    return new DefaultChatClientBuilder(chatModel);
  },
};
