import { expect, test } from 'bun:test';
import {
  AiError,
  AnthropicChatModel,
  type ChatModelProvider,
  createChatModel,
  DEFAULT_ANTHROPIC_MODEL,
  DEFAULT_OPENAI_MODEL,
  OpenAiChatModel,
  Prompt,
  SubscriptionChatModel,
} from '../src/index.ts';

const credentials = {
  OPENAI_API_KEY: 'openai',
  ANTHROPIC_API_KEY: 'anthropic',
  XAI_API_KEY: 'xai',
};
for (const [name, cli] of [
  ['openai', 'codex'],
  ['anthropic', 'claude'],
  ['xai', 'grok'],
  ['agy', 'agy'],
  ['junie', 'junie'],
] as const) {
  test(`${name} selects subscription CLI and its native model default`, () => {
    const model = createChatModel({ provider: name, auth: 'subscription', env: {} });
    expect(model).toBeInstanceOf(SubscriptionChatModel);
    expect((model as SubscriptionChatModel).config.provider).toBe(cli);
    expect((model as SubscriptionChatModel).config.model).toBeUndefined();
  });
}
for (const provider of ['openai', 'anthropic', 'xai', 'codex', 'claude', 'grok'] as const) {
  test(`${provider} supports explicit API access including aliases`, () => {
    const model = createChatModel({ provider, auth: 'api', env: credentials, model: 'chosen' });
    expect(model).toBeInstanceOf(
      ['anthropic', 'claude'].includes(provider) ? AnthropicChatModel : OpenAiChatModel,
    );
    expect(model.options?.model).toBe('chosen');
    if (['xai', 'grok'].includes(provider)) {
      expect((model as OpenAiChatModel).options?.apiKey).toBe('xai');
      expect((model as OpenAiChatModel).options?.baseUrl).toBe('https://api.x.ai/v1');
    }
  });
}
test('aliases default to subscription; explicit selection precedes environment', () => {
  for (const provider of ['codex', 'claude', 'grok', 'agy', 'junie'] as const)
    expect(createChatModel({ provider, env: {} })).toBeInstanceOf(SubscriptionChatModel);
  const model = createChatModel({
    provider: 'openai',
    auth: 'api',
    model: 'explicit',
    env: { ...credentials, PROVIDER: 'anthropic', AUTH: 'subscription', MODEL: 'env' },
    api: { model: 'nested' },
  });
  expect(model).toBeInstanceOf(OpenAiChatModel);
  expect(model.options?.model).toBe('explicit');
  expect(
    createChatModel({ env: { PROVIDER: 'openai', AUTH: 'subscription', MODEL: 'env' } }),
  ).toBeInstanceOf(SubscriptionChatModel);
});

test('HTTP requests preserve model defaults and supplied transport', async () => {
  for (const provider of ['openai', 'anthropic'] as const) {
    let request: Record<string, unknown> = {};
    const model = createChatModel({
      provider,
      env: credentials,
      api: {
        fetch: async (_url, init) => {
          request = JSON.parse(String(init?.body));
          return Response.json(
            provider === 'openai'
              ? { choices: [{ message: { content: 'ok' } }] }
              : { content: [{ type: 'text', text: 'ok' }] },
          );
        },
      },
    });
    await model.call(new Prompt('hello'));
    expect(request.model).toBe(
      provider === 'openai' ? DEFAULT_OPENAI_MODEL : DEFAULT_ANTHROPIC_MODEL,
    );
  }
});

test('Hermes uses only the existing subscription proxy', () => {
  const model = createChatModel({
    provider: 'hermes',
    model: 'portal',
    env: {},
  }) as OpenAiChatModel;
  expect(model.options?.baseUrl).toBe('http://127.0.0.1:8645/v1');
  expect(model.options?.apiKey).toBe('unused-local-proxy');
  expect(() => createChatModel({ provider: 'hermes', env: {} })).toThrow('requires MODEL');
});
test('invalid combinations fail without API fallback', () => {
  for (const provider of ['agy', 'junie', 'hermes'] as const)
    expect(() => createChatModel({ provider, auth: 'api', env: {} })).toThrow(AiError);
  for (const provider of ['xai', 'hermes'] as const)
    expect(() => createChatModel({ provider, env: credentials })).toThrow(AiError);
  expect(() => createChatModel({ env: {} })).toThrow('provider is required');
  expect(() => createChatModel({ provider: 'bad' as ChatModelProvider })).toThrow(
    'Unknown PROVIDER',
  );
  expect(() => createChatModel({ env: { PROVIDER: 'openai', AUTH: 'bad' } })).toThrow(
    'Unsupported AUTH',
  );
  expect(() =>
    createChatModel({
      provider: 'openai',
      auth: 'subscription',
      env: {},
      api: { apiKey: 'secret' },
    }),
  ).toThrow('API options');
  expect(() => createChatModel({ provider: 'openai', env: credentials, subscription: {} })).toThrow(
    'Subscription options',
  );
  expect(() =>
    createChatModel({ provider: 'xai', model: 'grok', env: { OPENAI_API_KEY: 'wrong' } }),
  ).toThrow('requires XAI_API_KEY');
  for (const [provider, key] of [
    ['openai', 'OPENAI_API_KEY'],
    ['anthropic', 'ANTHROPIC_AUTH_TOKEN'],
    ['xai', 'XAI_API_KEY'],
    ['agy', 'GEMINI_API_KEY'],
  ] as const)
    expect(() =>
      createChatModel({ provider, auth: 'subscription', env: { [key]: 'secret' } }),
    ).toThrow(key);
});

test('no-argument selection reads the environment', () => {
  const previous = {
    PROVIDER: process.env.PROVIDER,
    AUTH: process.env.AUTH,
    MODEL: process.env.MODEL,
  };
  try {
    process.env.PROVIDER = 'openai';
    process.env.AUTH = 'subscription';
    process.env.MODEL = 'native-override';
    const model = createChatModel() as SubscriptionChatModel;
    expect(model.config.provider).toBe('codex');
    expect(model.config.model).toBe('native-override');
  } finally {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
});

test('API provider-specific options and CLI options are rejected when unsupported', () => {
  expect(() =>
    createChatModel({ provider: 'openai', env: credentials, api: { messagesPath: '/wrong' } }),
  ).toThrow('Unsupported openai API option');
  expect(() => new SubscriptionChatModel({ provider: 'codex', maxCalls: 0 })).toThrow('maxCalls');
  expect(() => new SubscriptionChatModel({ provider: 'codex', timeoutMs: -1 })).toThrow(
    'timeoutMs',
  );
});
