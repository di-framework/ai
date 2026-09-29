import type { SecurityRequirement, SecurityScheme } from './types.ts';

export interface A2ABearerAuthOptions {
  readonly scheme?: string;
  readonly bearerFormat?: string;
  readonly description?: string;
  readonly validate?: (
    token: string,
    request: Request,
  ) => Promise<boolean | object | null | undefined> | boolean | object | null | undefined;
}

export interface A2AAuthOptions {
  readonly bearer?:
    | A2ABearerAuthOptions
    | ((token: string, request: Request) => Promise<boolean> | boolean);
  readonly customValidator?: (
    request: Request,
  ) => Promise<boolean | Response | null | undefined> | boolean | Response | null | undefined;
}

/**
 * Creates standard HTTP Bearer security scheme declarations for Agent Cards.
 */
export function createBearerSecurityScheme(
  options: { description?: string; bearerFormat?: string } = {},
): {
  securitySchemes: Record<string, SecurityScheme>;
  securityRequirements: SecurityRequirement[];
} {
  return {
    securitySchemes: {
      bearerAuth: {
        type: 'http',
        scheme: 'bearer',
        bearerFormat: options.bearerFormat ?? 'JWT',
        description: options.description ?? 'A2A Bearer Token Authentication',
      },
    },
    securityRequirements: [{ bearerAuth: [] }],
  };
}

/**
 * Creates an HTTP authentication hook for createA2AHttpHandler.
 * Fails closed with generic 401 error to prevent security policy leakage.
 */
export function createA2AAuthHandler(
  options: A2AAuthOptions,
): (request: Request) => Promise<boolean | Response> {
  return async (request: Request): Promise<boolean | Response> => {
    if (options.customValidator) {
      const customRes = await options.customValidator(request);
      if (customRes instanceof Response) return customRes;
      if (customRes === false) {
        return new Response(
          JSON.stringify({
            jsonrpc: '2.0',
            id: null,
            error: {
              code: -32001,
              message: 'Unauthorized: invalid or missing credentials',
            },
          }),
          {
            status: 401,
            headers: { 'Content-Type': 'application/json' },
          },
        );
      }
      if (customRes === true) return true;
    }

    if (options.bearer) {
      const authHeader =
        request.headers.get('Authorization') ?? request.headers.get('authorization');
      if (!authHeader || !authHeader.toLowerCase().startsWith('bearer')) {
        return new Response(
          JSON.stringify({
            jsonrpc: '2.0',
            id: null,
            error: {
              code: -32001,
              message: 'Unauthorized: missing Bearer token',
            },
          }),
          {
            status: 401,
            headers: {
              'Content-Type': 'application/json',
              'WWW-Authenticate': 'Bearer realm="A2A"',
            },
          },
        );
      }

      const token = authHeader.toLowerCase().startsWith('bearer')
        ? authHeader.slice(6).trim()
        : authHeader.trim();
      if (!token) {
        return new Response(
          JSON.stringify({
            jsonrpc: '2.0',
            id: null,
            error: {
              code: -32001,
              message: 'Unauthorized: empty Bearer token',
            },
          }),
          {
            status: 401,
            headers: { 'Content-Type': 'application/json' },
          },
        );
      }

      let valid: boolean | object | null | undefined = false;
      if (typeof options.bearer === 'function') {
        valid = await options.bearer(token, request);
      } else if (options.bearer.validate) {
        valid = await options.bearer.validate(token, request);
      } else {
        // Default: token present
        valid = token.length > 0;
      }

      if (!valid) {
        return new Response(
          JSON.stringify({
            jsonrpc: '2.0',
            id: null,
            error: {
              code: -32001,
              message: 'Unauthorized: invalid credentials',
            },
          }),
          {
            status: 401,
            headers: { 'Content-Type': 'application/json' },
          },
        );
      }

      return true;
    }

    return true;
  };
}

/**
 * Helper to build dynamic Authorization headers for A2AClient from a token source.
 */
export function createBearerCredentialSource(
  tokenOrProvider: string | (() => string | Promise<string>),
): () => Promise<Record<string, string>> {
  return async () => {
    const token = typeof tokenOrProvider === 'function' ? await tokenOrProvider() : tokenOrProvider;
    return {
      Authorization: `Bearer ${token}`,
    };
  };
}
