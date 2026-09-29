import { describe, expect, it } from 'bun:test';
import {
  type A2AAgentExecutor,
  A2AClient,
  AgentCardHelper,
  createA2AAuthHandler,
  createA2AHttpHandler,
  createBearerCredentialSource,
  createBearerSecurityScheme,
  createTextMessage,
} from '../../src/a2a/index.ts';

describe('A2A Authentication & Authorization', () => {
  const { securitySchemes, securityRequirements } = createBearerSecurityScheme({
    description: 'OAuth2 / Bearer JWT for SecuredAgent',
  });

  const securedCard = AgentCardHelper.create({
    name: 'SecuredAgent',
    description: 'Requires valid JWT bearer token',
    a2a: { url: 'http://secured.local/rpc' },
    skills: [{ id: 'vault.read', description: 'Read secure vault' }],
    security_schemes: securitySchemes,
    security_requirements: securityRequirements,
  });

  const secretExecutor: A2AAgentExecutor = {
    async execute(_task, _message, _ctx) {
      return {
        status: { state: 'completed' },
        messages: [createTextMessage('Vault secret: 42', 'agent')],
      };
    },
  };

  const authHandler = createA2AAuthHandler({
    bearer: {
      validate: (token) => token === 'valid-secret-token-xyz',
    },
  });

  const securedHttpHandler = createA2AHttpHandler({
    card: securedCard,
    executor: secretExecutor,
    authHandler,
  });

  const loopback = async (input: string | URL, init?: RequestInit): Promise<Response> => {
    const urlStr = typeof input === 'string' ? input : input.toString();
    return securedHttpHandler(new Request(urlStr, init));
  };

  it('serves Agent Card with declared security schemes and requirements without auth', async () => {
    const client = A2AClient.create({
      baseUrl: 'http://secured.local',
      fetch: loopback,
    });

    const card = await client.getCard();
    expect(card.security_schemes?.bearerAuth?.type).toBe('http');
    expect(card.security_schemes?.bearerAuth?.scheme).toBe('bearer');
    expect(card.security_requirements?.[0]?.bearerAuth).toBeDefined();
  });

  it('rejects unauthenticated requests with generic 401', async () => {
    const unauthenticatedClient = A2AClient.create({
      baseUrl: 'http://secured.local',
      fetch: loopback,
    });

    expect(
      unauthenticatedClient.send({
        skill: 'vault.read',
        message: 'Get secret',
      }),
    ).rejects.toThrow();
  });

  it('rejects invalid credentials without leaking internal policies', async () => {
    const badClient = A2AClient.create({
      baseUrl: 'http://secured.local',
      fetch: loopback,
      headers: createBearerCredentialSource('invalid-token-123'),
    });

    expect(
      badClient.send({
        skill: 'vault.read',
        message: 'Get secret',
      }),
    ).rejects.toThrow();
  });

  it('allows authenticated requests with valid bearer credentials from client credential source', async () => {
    const authorizedClient = A2AClient.create({
      baseUrl: 'http://secured.local',
      fetch: loopback,
      headers: createBearerCredentialSource(async () => 'valid-secret-token-xyz'),
    });

    const result = await authorizedClient.send({
      skill: 'vault.read',
      message: 'Get secret',
    });

    expect(result.task?.status.state).toBe('completed');
    const msg = result.task?.history?.[1]?.parts[0] as { text: string } | undefined;
    expect(msg?.text).toBe('Vault secret: 42');
  });

  it('supports customValidator hook with Response, false, and true', async () => {
    const customResHandler = createA2AAuthHandler({
      customValidator: () => new Response('Custom Block', { status: 403 }),
    });
    const res1 = await customResHandler(new Request('http://localhost'));
    expect(res1 instanceof Response).toBe(true);

    const customFalseHandler = createA2AAuthHandler({
      customValidator: () => false,
    });
    const res2 = await customFalseHandler(new Request('http://localhost'));
    expect(res2 instanceof Response).toBe(true);
    if (res2 instanceof Response) {
      expect(res2.status).toBe(401);
    }

    const customTrueHandler = createA2AAuthHandler({
      customValidator: () => true,
    });
    const res3 = await customTrueHandler(new Request('http://localhost'));
    expect(res3).toBe(true);

    const emptyBearerHandler = createA2AAuthHandler({
      bearer: (token) => token === 'abc',
    });
    const res4 = await emptyBearerHandler(
      new Request('http://localhost', { headers: { Authorization: 'Bearer ' } }),
    );
    expect(res4 instanceof Response).toBe(true);

    const defaultBearerHandler = createA2AAuthHandler({
      bearer: {},
    });
    const res5 = await defaultBearerHandler(
      new Request('http://localhost', { headers: { Authorization: 'Bearer anytoken' } }),
    );
    expect(res5).toBe(true);

    const credSource = createBearerCredentialSource('static-token');
    const headers = await credSource();
    expect(headers.Authorization).toBe('Bearer static-token');

    const fnBearerHandler = createA2AAuthHandler({
      bearer: (token) => token === 'direct-func',
    });
    const fnRes = await fnBearerHandler(
      new Request('http://localhost', { headers: { Authorization: 'Bearer direct-func' } }),
    );
    expect(fnRes).toBe(true);

    const emptyOptionsHandler = createA2AAuthHandler({});
    const emptyRes = await emptyOptionsHandler(new Request('http://localhost'));
    expect(emptyRes).toBe(true);
  });

  it('unsecured agents without security requirements work without credentials', async () => {
    const openCard = AgentCardHelper.create({
      name: 'OpenAgent',
      a2a: { url: 'http://open.local/rpc' },
      skills: [{ id: 'public.info', description: 'Public info' }],
    });

    const openHandler = createA2AHttpHandler({
      card: openCard,
      executor: {
        async execute() {
          return { status: { state: 'completed' } };
        },
      },
    });

    const openClient = A2AClient.create({
      baseUrl: 'http://open.local',
      fetch: async (input: string | URL, init?: RequestInit) => {
        return openHandler(new Request(typeof input === 'string' ? input : input.toString(), init));
      },
    });

    const res = await openClient.send({ message: 'Hello' });
    expect(res.task?.status.state).toBe('completed');
  });
});
