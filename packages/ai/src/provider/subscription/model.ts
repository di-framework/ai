import type { ChatModel } from '../../chat/model/chat-model.ts';
import type { ChatResponse } from '../../chat/model/chat-response.ts';
import { CALL_WITH_TOOL_MANAGER } from '../../chat/model/tool-execution-context.ts';
import type { Prompt } from '../../chat/prompt/prompt.ts';
import { AiError } from '../../model/errors.ts';
import type { ToolCallingManager } from '../../model/tool/tool-calling-manager.ts';
import type { BridgeProvider } from './launch.ts';
import type { BridgeEvent } from './server.ts';

export interface SubscriptionChatModelOptions {
  provider: BridgeProvider;
  model?: string;
  timeoutMs?: number;
  maxCalls?: number;
  toolCallingManager?: ToolCallingManager;
  onEvent?: (event: BridgeEvent) => void;
  executable?: string;
  /** Overrides inherited process environment for the CLI. Authentication conflicts are rejected. */
  env?: Readonly<Record<string, string | undefined>>;
}

const conflicts: Record<BridgeProvider | 'hermes', readonly string[]> = {
  codex: ['OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL'],
  claude: [
    'ANTHROPIC_API_KEY',
    'ANTHROPIC_AUTH_TOKEN',
    'ANTHROPIC_BASE_URL',
    'CLAUDE_CODE_USE_BEDROCK',
    'CLAUDE_CODE_USE_VERTEX',
    'CLAUDE_CODE_USE_FOUNDRY',
  ],
  grok: ['XAI_API_KEY', 'GROK_API_KEY', 'XAI_BASE_URL'],
  agy: [
    'GEMINI_API_KEY',
    'GOOGLE_API_KEY',
    'GOOGLE_APPLICATION_CREDENTIALS',
    'GOOGLE_GENAI_USE_VERTEXAI',
  ],
  junie: ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY'],
  hermes: ['NOUS_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY'],
};

export function assertSubscriptionEnvironment(
  provider: BridgeProvider | 'hermes',
  env: Readonly<Record<string, string | undefined>>,
): void {
  for (const key of conflicts[provider]) {
    if (env[key]?.trim()) {
      throw new AiError(
        `Subscription access conflicts with ${key}; use native subscription login`,
        'invalid-request',
      );
    }
  }
}

/** Native CLI loop; Bun is loaded only when inference is invoked. */
export class SubscriptionChatModel implements ChatModel {
  readonly config: Readonly<SubscriptionChatModelOptions>;

  constructor(config: SubscriptionChatModelOptions) {
    if (!Object.hasOwn(conflicts, config.provider) || (config.provider as string) === 'hermes') {
      throw new AiError('Unsupported subscription CLI provider', 'invalid-request');
    }
    const allowed = [
      'provider',
      'model',
      'timeoutMs',
      'maxCalls',
      'toolCallingManager',
      'onEvent',
      'executable',
      'env',
    ];
    for (const key of Object.keys(config)) {
      if (!allowed.includes(key))
        throw new AiError(`Unsupported subscription option: ${key}`, 'invalid-request');
    }
    for (const key of ['timeoutMs', 'maxCalls'] as const) {
      const value = config[key];
      if (value !== undefined && (!Number.isSafeInteger(value) || value < 1))
        throw new AiError(`${key} must be positive`, 'invalid-request');
    }
    this.config = { ...config, env: config.env ? { ...config.env } : undefined };
    this.checkEnvironment();
  }

  private checkEnvironment(): void {
    assertSubscriptionEnvironment(this.config.provider, {
      ...(typeof process === 'undefined' ? {} : process.env),
      ...this.config.env,
    });
  }

  call(prompt: Prompt): Promise<ChatResponse> {
    return this[CALL_WITH_TOOL_MANAGER](prompt);
  }

  async [CALL_WITH_TOOL_MANAGER](
    prompt: Prompt,
    manager?: ToolCallingManager,
  ): Promise<ChatResponse> {
    try {
      this.checkEnvironment();
      if (prompt.options?.signal?.aborted)
        throw new AiError('Subscription request cancelled', 'cancelled');
      if (typeof Bun === 'undefined')
        throw new AiError('Subscription CLI inference requires Bun', 'invalid-request');
      const { callSubscription } = await import('./runtime.ts');
      return await callSubscription(
        { ...this.config, toolCallingManager: manager ?? this.config.toolCallingManager },
        prompt,
      );
    } catch (cause) {
      if (cause instanceof AiError) throw cause;
      throw new AiError(
        cause instanceof Error ? cause.message : 'Subscription CLI failed',
        'provider-error',
        {
          provider: this.config.provider,
          cause,
          retryable: false,
        },
      );
    }
  }
}
