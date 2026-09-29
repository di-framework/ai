import type { AuthContainer, AuthorizationManager, Principal } from '@di-framework/auth';
import { resolveAuthorizationManager } from '@di-framework/auth';
import { getToolMetadata } from '../../tool/tool-callback.ts';
import type { ToolContext } from '../../tool/tool-context.ts';
import type {
  ToolExecutionAdvisor,
  ToolExecutionAdvisorContext,
  ToolExecutionAdvisorNext,
} from './tool-execution-advisor.ts';

/**
 * Authorization context passed to {@link AuthorizationManager} for AI tool authorization.
 */
export interface ToolAuthorizationContext<TMetadata = unknown> {
  readonly transport: 'ai-tool';
  readonly tool: string;
  readonly arguments: unknown;
  readonly context: ToolContext;
  readonly metadata?: TMetadata;
}

/**
 * Convenient alias for an {@link AuthorizationManager} specialized for AI tool authorization context.
 */
export type ToolAuthorizationManager = AuthorizationManager<ToolAuthorizationContext>;

/**
 * Resolver function extracting an authenticated {@link Principal} from trusted {@link ToolContext}.
 */
export type PrincipalResolver = (context: ToolContext) => Principal | undefined;

/**
 * Default principal resolver reading `principal` or `user` from trusted {@link ToolContext}.
 * Model-generated arguments are NEVER used to resolve or overwrite the principal.
 */
export function defaultPrincipalResolver(toolContext: ToolContext): Principal | undefined {
  const p = toolContext.get('principal') ?? toolContext.get('user');
  if (p && typeof p === 'object') {
    return p as Principal;
  }
  return undefined;
}

export interface ToolAuthorizationAdvisorOptions {
  /** Explicit AuthorizationManager instance or factory. */
  readonly authorizationManager?:
    | AuthorizationManager<ToolAuthorizationContext>
    | (() => AuthorizationManager<ToolAuthorizationContext>);
  /** Alias for `authorizationManager`. */
  readonly manager?:
    | AuthorizationManager<ToolAuthorizationContext>
    | (() => AuthorizationManager<ToolAuthorizationContext>);
  /** DI Token used to resolve AuthorizationManager when omitted. */
  readonly managerToken?: string;
  /** Auth container instance for DI resolution. */
  readonly container?: AuthContainer;
  /** Function to resolve trusted Principal from ToolContext. Defaults to `defaultPrincipalResolver`. */
  readonly principalResolver?: PrincipalResolver;
  /** Generic unauthorized message returned to model when denied/failed. Defaults to "Tool execution unauthorized". */
  readonly unauthorizedMessage?: string;
  /** Advisor order in execution chain (default: 0). */
  readonly order?: number;
  /** Advisor name. */
  readonly name?: string;
}

/**
 * {@link ToolExecutionAdvisor} integrating tool execution authorization via `@di-framework/auth` {@link AuthorizationManager}.
 */
export class ToolAuthorizationAdvisor implements ToolExecutionAdvisor {
  readonly name: string;
  readonly order: number;
  private readonly options: ToolAuthorizationAdvisorOptions;

  constructor(options: ToolAuthorizationAdvisorOptions = {}) {
    this.options = options;
    this.name = options.name ?? 'Tool Authorization Advisor';
    this.order = options.order ?? 0;
  }

  async adviseExecution(
    context: ToolExecutionAdvisorContext,
    next: ToolExecutionAdvisorNext,
  ): Promise<string> {
    const unauthorizedMessage = this.options.unauthorizedMessage ?? 'Tool execution unauthorized';

    // 1. Resolve Principal ONLY from trusted ToolContext (never model-generated arguments)
    const principalResolver = this.options.principalResolver ?? defaultPrincipalResolver;
    let principal: Principal | undefined;
    try {
      principal = principalResolver(context.toolContext);
    } catch {
      return unauthorizedMessage;
    }

    if (!principal) {
      return unauthorizedMessage;
    }

    // 2. Resolve AuthorizationManager
    let manager: AuthorizationManager<ToolAuthorizationContext> | undefined;
    try {
      manager = this.resolveManager();
    } catch {
      return unauthorizedMessage;
    }

    if (!manager) {
      return unauthorizedMessage;
    }

    // 3. Prepare authorization context
    let parsedArgs: unknown;
    const rawArgs = context.toolCall.arguments;
    if (!rawArgs || rawArgs.trim() === '') {
      parsedArgs = {};
    } else {
      try {
        parsedArgs = JSON.parse(rawArgs);
      } catch {
        parsedArgs = rawArgs;
      }
    }

    const toolMetadata = getToolMetadata(context.toolCallback);
    const metadata = toolMetadata.auth !== undefined ? toolMetadata.auth : toolMetadata;

    const authContext: ToolAuthorizationContext = {
      transport: 'ai-tool',
      tool: context.toolCall.name,
      arguments: parsedArgs,
      context: context.toolContext,
      metadata,
    };

    // 4. Evaluate authorization policy
    try {
      const result = await manager.authorize(principal, authContext);
      if (result && result.allowed === true) {
        return await next(context);
      }
    } catch {
      // Fail closed on error/exception
      return unauthorizedMessage;
    }

    // Fail closed on denial
    return unauthorizedMessage;
  }

  private resolveManager(): AuthorizationManager<ToolAuthorizationContext> | undefined {
    if (typeof this.options.authorizationManager === 'function') {
      return this.options.authorizationManager();
    }
    if (this.options.authorizationManager) {
      return this.options.authorizationManager;
    }
    if (typeof this.options.manager === 'function') {
      return this.options.manager();
    }
    if (this.options.manager) {
      return this.options.manager;
    }

    return resolveAuthorizationManager<ToolAuthorizationContext>({
      managerToken: this.options.managerToken,
      container: this.options.container,
    });
  }
}

/**
 * Factory function for creating a {@link ToolAuthorizationAdvisor}.
 */
export function toolAuthorizationAdvisor(
  options?: ToolAuthorizationAdvisorOptions,
): ToolAuthorizationAdvisor {
  return new ToolAuthorizationAdvisor(options);
}
