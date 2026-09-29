import { AiAnnKeys } from './keys.ts';
import {
  defineMethodAnn,
  defineOnCtor,
  defineParamAnn,
  readMethodAnnMap,
  readOnCtor,
} from './meta.ts';

export interface McpClientOptions {
  readonly token?: string;
  readonly title?: string;
}

/** Inject/adapt an MCP session as tool callbacks. */
export function McpClient(
  options: McpClientOptions | string = {},
): ClassDecorator & ParameterDecorator & PropertyDecorator {
  const opts: McpClientOptions = typeof options === 'string' ? { token: options } : options;
  return ((
    target: object,
    propertyKey?: string | symbol,
    indexOrDescriptor?: number | PropertyDescriptor,
  ) => {
    if (typeof indexOrDescriptor === 'number') {
      defineParamAnn(
        AiAnnKeys.MCP_CLIENT,
        target,
        propertyKey !== undefined ? String(propertyKey) : undefined,
        indexOrDescriptor,
        opts,
      );
      return;
    }
    if (propertyKey !== undefined) {
      defineOnCtor(`${AiAnnKeys.MCP_CLIENT}:prop:${String(propertyKey)}`, opts, target);
      return;
    }
    defineOnCtor(AiAnnKeys.MCP_CLIENT, opts, target);
  }) as ClassDecorator & ParameterDecorator & PropertyDecorator;
}

export interface McpToolOptions {
  readonly name?: string;
  readonly description?: string;
}

/** Expose a local method as an MCP tool. */
export function McpTool(options: McpToolOptions | string = {}): MethodDecorator {
  const opts: McpToolOptions = typeof options === 'string' ? { description: options } : options;
  return (target, propertyKey) => {
    defineMethodAnn(AiAnnKeys.MCP_TOOL, target as object, String(propertyKey), opts);
  };
}

export function getMcpClientOptions(target: object): McpClientOptions | undefined {
  return readOnCtor(AiAnnKeys.MCP_CLIENT, target);
}

export function getMcpTools(target: object): Readonly<Record<string, McpToolOptions>> {
  return readMethodAnnMap<McpToolOptions>(AiAnnKeys.MCP_TOOL, target);
}
