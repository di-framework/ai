import { AiAnnKeys } from './keys.ts';
import { defineMethodAnn, defineOnCtor, readMethodAnnMap, readOnCtor } from './meta.ts';

export interface ChainOptions {
  readonly steps?: readonly string[];
}

function isReadonlyStringArray(value: unknown): value is readonly string[] {
  return Array.isArray(value);
}

/** Build a sequential ChainWorkflow from steps/methods. */
export function Chain(options: ChainOptions | readonly string[] = {}): ClassDecorator {
  const opts: ChainOptions = isReadonlyStringArray(options) ? { steps: options } : options;
  return (target) => {
    defineOnCtor(AiAnnKeys.CHAIN, opts, target as object);
  };
}

export interface RouteOptions {
  readonly routes?: Readonly<Record<string, string>>;
}

/** Classify then dispatch (RoutingWorkflow). */
export function Route(options: RouteOptions = {}): ClassDecorator & MethodDecorator {
  return ((target: object, propertyKey?: string | symbol) => {
    if (propertyKey !== undefined) {
      defineMethodAnn(AiAnnKeys.ROUTE, target, String(propertyKey), options);
    } else {
      defineOnCtor(AiAnnKeys.ROUTE, options, target);
    }
  }) as ClassDecorator & MethodDecorator;
}

/** Alias of {@link Route}. */
export const Router = Route;

export interface ParallelOptions {
  readonly prompts?: readonly string[];
}

/** Fan-out concurrent prompts (ParallelizationWorkflow). */
export function Parallel(options: ParallelOptions | readonly string[] = {}): ClassDecorator {
  const opts: ParallelOptions = isReadonlyStringArray(options) ? { prompts: options } : options;
  return (target) => {
    defineOnCtor(AiAnnKeys.PARALLEL, opts, target as object);
  };
}

export interface OrchestratorOptions {
  readonly role?: string;
}

/** Plan/delegate role in orchestrator–workers workflow. */
export function Orchestrator(
  options: OrchestratorOptions | string = {},
): ClassDecorator & MethodDecorator {
  const opts: OrchestratorOptions = typeof options === 'string' ? { role: options } : options;
  return ((target: object, propertyKey?: string | symbol) => {
    if (propertyKey !== undefined) {
      defineMethodAnn(AiAnnKeys.ORCHESTRATOR, target, String(propertyKey), opts);
    } else {
      defineOnCtor(AiAnnKeys.ORCHESTRATOR, opts, target);
    }
  }) as ClassDecorator & MethodDecorator;
}

/** Worker role in orchestrator–workers workflow. */
export function Worker(
  options: OrchestratorOptions | string = {},
): ClassDecorator & MethodDecorator {
  const opts: OrchestratorOptions = typeof options === 'string' ? { role: options } : options;
  return ((target: object, propertyKey?: string | symbol) => {
    if (propertyKey !== undefined) {
      defineMethodAnn(AiAnnKeys.WORKER, target, String(propertyKey), opts);
    } else {
      defineOnCtor(AiAnnKeys.WORKER, opts, target);
    }
  }) as ClassDecorator & MethodDecorator;
}

export interface EvaluateOptions {
  readonly criteria?: string;
}

/** Evaluate step in evaluator–optimizer loop. */
export function Evaluate(options: EvaluateOptions | string = {}): MethodDecorator & ClassDecorator {
  const opts: EvaluateOptions = typeof options === 'string' ? { criteria: options } : options;
  return ((target: object, propertyKey?: string | symbol) => {
    if (propertyKey !== undefined) {
      defineMethodAnn(AiAnnKeys.EVALUATE, target, String(propertyKey), opts);
    } else {
      defineOnCtor(AiAnnKeys.EVALUATE, opts, target);
    }
  }) as MethodDecorator & ClassDecorator;
}

/** Optimize/refine step in evaluator–optimizer loop. */
export function Optimize(options: EvaluateOptions | string = {}): MethodDecorator & ClassDecorator {
  const opts: EvaluateOptions = typeof options === 'string' ? { criteria: options } : options;
  return ((target: object, propertyKey?: string | symbol) => {
    if (propertyKey !== undefined) {
      defineMethodAnn(AiAnnKeys.OPTIMIZE, target, String(propertyKey), opts);
    } else {
      defineOnCtor(AiAnnKeys.OPTIMIZE, opts, target);
    }
  }) as MethodDecorator & ClassDecorator;
}

export function getChainOptions(target: object): ChainOptions | undefined {
  return readOnCtor(AiAnnKeys.CHAIN, target);
}

export function getRouteOptions(target: object): RouteOptions | undefined {
  return readOnCtor(AiAnnKeys.ROUTE, target);
}

export function getParallelOptions(target: object): ParallelOptions | undefined {
  return readOnCtor(AiAnnKeys.PARALLEL, target);
}

export function getOrchestratorOptions(target: object): OrchestratorOptions | undefined {
  return readOnCtor(AiAnnKeys.ORCHESTRATOR, target);
}

export function getWorkerMethods(target: object): Readonly<Record<string, OrchestratorOptions>> {
  return readMethodAnnMap<OrchestratorOptions>(AiAnnKeys.WORKER, target);
}

export function getEvaluateOptions(target: object): EvaluateOptions | undefined {
  return readOnCtor(AiAnnKeys.EVALUATE, target);
}

export function getOptimizeOptions(target: object): EvaluateOptions | undefined {
  return readOnCtor(AiAnnKeys.OPTIMIZE, target);
}
