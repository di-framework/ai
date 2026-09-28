import type { AnyConstructor } from './keys.ts';
import { AiAnnKeys } from './keys.ts';
import { defineOnCtor, readOnCtor } from './meta.ts';

export interface AdvisorAnnOptions {
  readonly order?: number;
}

/** Register class as a chat advisor bean. */
export function Advisor(options: AdvisorAnnOptions = {}): ClassDecorator {
  return (target) => {
    defineOnCtor(AiAnnKeys.ADVISOR, options, target as object);
    if (options.order !== undefined) {
      defineOnCtor(AiAnnKeys.ADVISOR_ORDER, options.order, target as object);
    }
  };
}

/** Advisor precedence in the chain. */
export function AdvisorOrder(order: number): ClassDecorator {
  return (target) => {
    defineOnCtor(AiAnnKeys.ADVISOR_ORDER, order, target as object);
  };
}

/** Alias of {@link AdvisorOrder} when no conflict. */
export const Order = AdvisorOrder;

export interface WithMemoryOptions {
  readonly enabled?: boolean;
  readonly token?: string;
}

/** Attach chat-memory advisor to assistant/agent/client. */
export function WithMemory(options: WithMemoryOptions | boolean = true): ClassDecorator {
  const opts: WithMemoryOptions =
    typeof options === 'boolean' ? { enabled: options } : { enabled: true, ...options };
  return (target) => {
    defineOnCtor(AiAnnKeys.WITH_MEMORY, opts, target as object);
  };
}

export interface WithRagOptions {
  readonly topK?: number;
  readonly vectorStore?: string;
  readonly retriever?: string | AnyConstructor;
  readonly enabled?: boolean;
}

/** Attach retrieval-augmentation advisor. */
export function WithRag(options: WithRagOptions | boolean = true): ClassDecorator {
  const opts: WithRagOptions =
    typeof options === 'boolean' ? { enabled: options } : { enabled: true, ...options };
  return (target) => {
    defineOnCtor(AiAnnKeys.WITH_RAG, opts, target as object);
  };
}

/** Alias of {@link WithRag}. */
export const RetrievalAugmented = WithRag;

export interface WithToolsOptions {
  readonly tools?: readonly (AnyConstructor | object)[];
  readonly enabled?: boolean;
}

/** Attach tool-calling / tool sources. */
export function WithTools(
  toolsOrOptions: readonly (AnyConstructor | object)[] | WithToolsOptions | boolean = true,
): ClassDecorator {
  const opts: WithToolsOptions = Array.isArray(toolsOrOptions)
    ? { enabled: true, tools: toolsOrOptions }
    : typeof toolsOrOptions === 'boolean'
      ? { enabled: toolsOrOptions }
      : { enabled: true, ...toolsOrOptions };
  return (target) => {
    defineOnCtor(AiAnnKeys.WITH_TOOLS, opts, target as object);
  };
}

export interface AiObservedOptions {
  readonly includePromptText?: boolean;
  readonly includeResponseText?: boolean;
  readonly enabled?: boolean;
}

/** Emit observation/logging events for calls. */
export function AiObserved(options: AiObservedOptions | boolean = true): ClassDecorator {
  const opts: AiObservedOptions =
    typeof options === 'boolean' ? { enabled: options } : { enabled: true, ...options };
  return (target) => {
    defineOnCtor(AiAnnKeys.AI_OBSERVED, opts, target as object);
  };
}

/** Alias of {@link AiObserved}. */
export const Observed = AiObserved;

export function getAdvisorOptions(target: object): AdvisorAnnOptions | undefined {
  return readOnCtor(AiAnnKeys.ADVISOR, target);
}

export function getAdvisorOrder(target: object): number | undefined {
  return readOnCtor(AiAnnKeys.ADVISOR_ORDER, target);
}

export function getWithMemory(target: object): WithMemoryOptions | undefined {
  return readOnCtor(AiAnnKeys.WITH_MEMORY, target);
}

export function getWithRag(target: object): WithRagOptions | undefined {
  return readOnCtor(AiAnnKeys.WITH_RAG, target);
}

export function getWithTools(target: object): WithToolsOptions | undefined {
  return readOnCtor(AiAnnKeys.WITH_TOOLS, target);
}

export function getAiObserved(target: object): AiObservedOptions | undefined {
  return readOnCtor(AiAnnKeys.AI_OBSERVED, target);
}
