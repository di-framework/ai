import type { ToolSource } from '../../chat/client/default-chat-client.ts';
import type { AnyConstructor } from './keys.ts';
import { AiAnnKeys } from './keys.ts';
import {
  defineMethodAnn,
  defineOnCtor,
  defineParamAnn,
  readMethodAnnMap,
  readOnCtor,
  readParamAnnMap,
} from './meta.ts';

export interface AiServiceOptions {
  readonly tools?: readonly (AnyConstructor | object | ToolSource)[];
  readonly chatModel?: string;
  readonly chatClient?: string;
  readonly system?: string;
}

/** Declarative injectable chat assistant (abstract class → proxy). */
export function AiService(options: AiServiceOptions = {}): ClassDecorator {
  return (target) => {
    defineOnCtor(AiAnnKeys.AI_SERVICE, options, target as object);
  };
}

/** Alias of {@link AiService}. */
export const Assistant = AiService;

export interface AgentOptions {
  readonly name?: string;
  readonly description?: string;
  readonly version?: string;
  readonly system?: string;
  readonly tools?: readonly (AnyConstructor | object | ToolSource)[];
  readonly memory?: boolean;
  readonly maxIterations?: number;
  readonly chatModel?: string;
  readonly defaultConversationId?: string;
  readonly skills?: readonly import('../../a2a/types.ts').AgentSkill[];
  readonly a2a?: { readonly url: string; readonly binding?: 'JSONRPC' | 'HTTP' };
}

/** Declarative ChatAgent bean with tools/memory and optional A2A network interface. */
export function Agent(options: AgentOptions = {}): ClassDecorator {
  return (target) => {
    defineOnCtor(AiAnnKeys.AGENT, options, target as object);
  };
}

/**
 * Alias of {@link Agent}. Named {@code ChatAgentBean} to avoid colliding with
 * the {@code ChatAgent} class export.
 */
export const ChatAgentBean = Agent;

export function SystemMessage(text: string): MethodDecorator & ClassDecorator {
  return ((target: object, propertyKey?: string | symbol) => {
    if (propertyKey !== undefined) {
      defineMethodAnn(AiAnnKeys.SYSTEM_MESSAGE, target, String(propertyKey), text);
    } else {
      defineOnCtor(AiAnnKeys.SYSTEM_MESSAGE, text, target);
    }
  }) as MethodDecorator & ClassDecorator;
}

export interface UserMessageOptions {
  readonly template?: string;
  /** When true (default for params), the argument value is the user message. */
  readonly fromParam?: boolean;
}

/** User prompt template, or bind a parameter as user text. */
export function UserMessage(
  templateOrOptions: string | UserMessageOptions = {},
): MethodDecorator & ParameterDecorator {
  const opts: UserMessageOptions =
    typeof templateOrOptions === 'string' ? { template: templateOrOptions } : templateOrOptions;
  return ((
    target: object,
    propertyKey?: string | symbol,
    indexOrDescriptor?: number | PropertyDescriptor,
  ) => {
    if (typeof indexOrDescriptor === 'number') {
      defineParamAnn(
        AiAnnKeys.USER_MESSAGE,
        target,
        propertyKey !== undefined ? String(propertyKey) : undefined,
        indexOrDescriptor,
        { ...opts, fromParam: opts.fromParam !== false },
      );
      return;
    }
    if (propertyKey !== undefined) {
      defineMethodAnn(AiAnnKeys.USER_MESSAGE, target, String(propertyKey), opts);
    }
  }) as MethodDecorator & ParameterDecorator;
}

/** Fixed few-shot assistant turn in the prompt. */
export function AssistantMessage(text: string): MethodDecorator {
  return (target, propertyKey) => {
    defineMethodAnn(AiAnnKeys.ASSISTANT_MESSAGE, target as object, String(propertyKey), text);
  };
}

export interface PromptVariableOptions {
  readonly name?: string;
}

/** Bind method arg into prompt template placeholders. */
export function V(nameOrOptions: string | PromptVariableOptions = {}): ParameterDecorator {
  const opts: PromptVariableOptions =
    typeof nameOrOptions === 'string' ? { name: nameOrOptions } : nameOrOptions;
  return (target, propertyKey, parameterIndex) => {
    defineParamAnn(
      AiAnnKeys.PROMPT_VARIABLE,
      target as object,
      propertyKey !== undefined ? String(propertyKey) : undefined,
      parameterIndex,
      opts,
    );
  };
}

/** Alias of {@link V}. */
export const PromptVariable = V;

/** Conversation/session id for chat memory. */
export function MemoryId(): ParameterDecorator {
  return (target, propertyKey, parameterIndex) => {
    defineParamAnn(
      AiAnnKeys.MEMORY_ID,
      target as object,
      propertyKey !== undefined ? String(propertyKey) : undefined,
      parameterIndex,
      true,
    );
  };
}

/** Alias of {@link MemoryId}. */
export const ConversationId = MemoryId;

export type AgentStrategyKind =
  | 'chat'
  | 'chain'
  | 'route'
  | 'parallel'
  | 'orchestrator'
  | 'evaluate';

export interface AgentStrategyOptions {
  readonly kind?: AgentStrategyKind;
  readonly steps?: readonly string[];
}

/** Attach a workflow/strategy to the agent. */
export function AgentStrategy(
  options: AgentStrategyOptions | AgentStrategyKind = {},
): ClassDecorator {
  const opts: AgentStrategyOptions = typeof options === 'string' ? { kind: options } : options;
  return (target) => {
    defineOnCtor(AiAnnKeys.AGENT_STRATEGY, opts, target as object);
  };
}

/** Cap tool-calling / agent loop rounds. */
export function MaxIterations(n: number): ClassDecorator {
  return (target) => {
    defineOnCtor(AiAnnKeys.MAX_ITERATIONS, n, target as object);
  };
}

export function getAiServiceOptions(target: object): AiServiceOptions | undefined {
  return readOnCtor(AiAnnKeys.AI_SERVICE, target);
}

export function getAgentOptions(target: object): AgentOptions | undefined {
  return readOnCtor(AiAnnKeys.AGENT, target);
}

export function getSystemMessage(target: object, methodName?: string): string | undefined {
  if (methodName) {
    const map = readMethodAnnMap<string>(AiAnnKeys.SYSTEM_MESSAGE, target);
    if (map[methodName]) return map[methodName];
  }
  const classLevel = readOnCtor<string>(AiAnnKeys.SYSTEM_MESSAGE, target);
  return typeof classLevel === 'string' ? classLevel : undefined;
}

export function getUserMessageAnn(
  target: object,
  methodName: string,
): UserMessageOptions | undefined {
  return readMethodAnnMap<UserMessageOptions>(AiAnnKeys.USER_MESSAGE, target)[methodName];
}

export function getAssistantMessage(target: object, methodName: string): string | undefined {
  return readMethodAnnMap<string>(AiAnnKeys.ASSISTANT_MESSAGE, target)[methodName];
}

export function getMemoryIdParams(
  target: object,
  methodName: string,
): readonly { index: number }[] {
  return readParamAnnMap<boolean>(AiAnnKeys.MEMORY_ID, target)[methodName] ?? [];
}

export function getUserMessageParams(
  target: object,
  methodName: string,
): readonly { index: number; value: UserMessageOptions }[] {
  return readParamAnnMap<UserMessageOptions>(AiAnnKeys.USER_MESSAGE, target)[methodName] ?? [];
}

export function getPromptVariables(
  target: object,
  methodName: string,
): readonly { index: number; value: PromptVariableOptions }[] {
  return (
    readParamAnnMap<PromptVariableOptions>(AiAnnKeys.PROMPT_VARIABLE, target)[methodName] ?? []
  );
}

export function getAgentStrategy(target: object): AgentStrategyOptions | undefined {
  return readOnCtor(AiAnnKeys.AGENT_STRATEGY, target);
}

export function getMaxIterations(target: object): number | undefined {
  return readOnCtor(AiAnnKeys.MAX_ITERATIONS, target);
}
