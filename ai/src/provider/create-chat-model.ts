import type { ChatModel } from '../chat/model/chat-model.ts';
import { AiError } from '../model/errors.ts';
import { AnthropicChatModel } from './anthropic/anthropic-chat-model.ts';
import type { AnthropicChatOptions } from './anthropic/anthropic-chat-options.ts';
import { OpenAiChatModel } from './openai/openai-chat-model.ts';
import type { OpenAiChatOptions } from './openai/openai-chat-options.ts';
import {
  assertSubscriptionEnvironment,
  SubscriptionChatModel,
  type SubscriptionChatModelOptions,
} from './subscription/model.ts';

export type ChatModelProvider =
  | 'openai'
  | 'anthropic'
  | 'xai'
  | 'agy'
  | 'junie'
  | 'hermes'
  | 'codex'
  | 'claude'
  | 'grok';
export type ChatModelAuth = 'api' | 'subscription';
export interface CreateChatModelOptions {
  provider?: ChatModelProvider;
  auth?: ChatModelAuth;
  model?: string;
  env?: Readonly<Record<string, string | undefined>>;
  api?: OpenAiChatOptions | AnthropicChatOptions;
  subscription?: Omit<SubscriptionChatModelOptions, 'provider' | 'model' | 'env'>;
}

/** Synchronous selection; never switches authentication routes after a failure. */
export function createChatModel(config: CreateChatModelOptions = {}): ChatModel {
  const env = config.env ?? (typeof process === 'undefined' ? {} : process.env);
  const fail = (message: string): never => {
    throw new AiError(message, 'invalid-request');
  };
  for (const key of Object.keys(config)) {
    if (!['provider', 'auth', 'model', 'env', 'api', 'subscription'].includes(key))
      fail(`Unsupported createChatModel option: ${key}`);
  }
  const name = (config.provider ?? env.PROVIDER ?? '').trim().toLowerCase();
  if (!name) fail('A provider is required; set provider or PROVIDER');
  const aliases: Record<string, string> = { codex: 'openai', claude: 'anthropic', grok: 'xai' };
  const provider = Object.hasOwn(aliases, name) ? (aliases[name] ?? name) : name;
  if (!['openai', 'anthropic', 'xai', 'agy', 'junie', 'hermes'].includes(provider))
    fail(`Unknown PROVIDER=${name}`);
  const auth = (
    config.auth ??
    env.AUTH ??
    (Object.hasOwn(aliases, name) || ['agy', 'junie', 'hermes'].includes(provider)
      ? 'subscription'
      : 'api')
  )
    .trim()
    .toLowerCase();
  if (auth !== 'api' && auth !== 'subscription') fail(`Unsupported AUTH=${auth}`);
  const model = config.model ?? config.api?.model ?? env.MODEL?.trim();
  if (auth === 'subscription') {
    if (config.api !== undefined) fail('API options are not supported with subscription access');
    const cli =
      provider === 'openai'
        ? 'codex'
        : provider === 'anthropic'
          ? 'claude'
          : provider === 'xai'
            ? 'grok'
            : (provider as 'agy' | 'junie' | 'hermes');
    assertSubscriptionEnvironment(cli, {
      ...(typeof process === 'undefined' ? {} : process.env),
      ...env,
    });
    if (cli === 'hermes') {
      if (config.subscription !== undefined)
        fail('Hermes proxy does not support CLI subscription options');
      if (!model) fail('Hermes requires MODEL or model');
      return new OpenAiChatModel({
        model,
        baseUrl: 'http://127.0.0.1:8645/v1',
        apiKey: 'unused-local-proxy',
      });
    }
    return new SubscriptionChatModel({ ...config.subscription, provider: cli, model, env });
  }
  if (config.subscription !== undefined)
    fail('Subscription options are not supported with API access');
  if (!['openai', 'anthropic', 'xai'].includes(provider))
    fail(`Provider ${provider} does not support API access`);
  if (provider === 'xai' && !model) fail('xAI API requires MODEL or model');
  const key =
    provider === 'anthropic'
      ? 'ANTHROPIC_API_KEY'
      : provider === 'xai'
        ? 'XAI_API_KEY'
        : 'OPENAI_API_KEY';
  const apiKey = config.api?.apiKey ?? env[key];
  if (!apiKey?.trim()) fail(`PROVIDER=${provider} requires ${key}`);
  const common = [
    'model',
    'apiKey',
    'baseUrl',
    'headers',
    'fetch',
    'signal',
    'temperature',
    'topP',
    'topK',
    'frequencyPenalty',
    'presencePenalty',
    'maxTokens',
    'stopSequences',
    'toolCallbacks',
    'toolContext',
    'outputSchema',
    'providerOptions',
  ];
  const native =
    provider === 'anthropic'
      ? ['messagesPath', 'anthropicVersion', 'defaultMaxTokens']
      : ['completionsPath', 'organization', 'project', 'useMaxCompletionTokens'];
  for (const key of Object.keys(config.api ?? {})) {
    if (![...common, ...native].includes(key)) fail(`Unsupported ${provider} API option: ${key}`);
  }
  const options = { ...config.api, model, apiKey };
  return provider === 'anthropic'
    ? new AnthropicChatModel(options)
    : new OpenAiChatModel({
        ...(provider === 'xai' ? { baseUrl: 'https://api.x.ai/v1' } : {}),
        ...options,
      });
}
