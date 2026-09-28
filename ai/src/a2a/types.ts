/**
 * Agent-to-Agent (A2A) Protocol 1.0 Types.
 *
 * ## SDK Decision
 * First-party types aligned directly with the normative A2A 1.0 protocol specification
 * (specification/a2a.proto) and REST/JSON-RPC wire representations.
 *
 * We implement first-party types and standard `fetch` over `@a2a-js/sdk` to avoid
 * Node-only dependencies (such as gRPC or Node native streams) and ensure zero-friction
 * runtime portability across Bun, Node, Deno, and Cloudflare Workers (the same portability
 * rule followed across @di-framework/ai chat and model providers).
 *
 * Opacity constraint:
 * Remote callers interact strictly through AgentCard, A2ATask, Messages, and Artifacts.
 * Internal model prompts, MCP tools, and memory identifiers are NEVER exposed across the wire.
 */

/** Protocol version for A2A 1.0 */
export const A2A_PROTOCOL_VERSION = '1.0';

/** Well-known relative path for Agent Card discovery */
export const AGENT_CARD_WELL_KNOWN_PATH = '/.well-known/agent-card.json';

/** JSON-RPC 2.0 A2A PascalCase method names */
export const A2AMethods = {
  SEND_MESSAGE: 'SendMessage',
  GET_TASK: 'GetTask',
  LIST_TASKS: 'ListTasks',
  CANCEL_TASK: 'CancelTask',
} as const;

export type A2AMethod = (typeof A2AMethods)[keyof typeof A2AMethods];

/**
 * Valid lifecycle states for an A2A Task.
 */
export type TaskState =
  | 'submitted'
  | 'working'
  | 'input-required'
  | 'auth-required'
  | 'completed'
  | 'failed'
  | 'canceled'
  | 'rejected';

/** Terminal task states that reject further state transitions */
export const TERMINAL_TASK_STATES: readonly TaskState[] = [
  'completed',
  'failed',
  'canceled',
  'rejected',
] as const;

export function isTerminalTaskState(state: TaskState): boolean {
  return (TERMINAL_TASK_STATES as readonly string[]).includes(state);
}

/** Task status descriptor */
export interface TaskStatus {
  readonly state: TaskState;
  readonly message?: A2AMessage;
  readonly timestamp?: string;
}

/** Participant role in an A2A interaction */
export type Role = 'user' | 'agent';

/** Text part in a message or artifact */
export interface TextPart {
  readonly kind: 'text';
  readonly text: string;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

/** File or blob part in a message or artifact */
export interface FilePart {
  readonly kind: 'file';
  readonly uri?: string;
  readonly mimeType?: string;
  readonly bytes?: string; // base64 encoded
  readonly name?: string;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Structured data part in a message or artifact */
export interface DataPart {
  readonly kind: 'data';
  readonly data: Readonly<Record<string, unknown>>;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Generic part union */
export type Part =
  | TextPart
  | FilePart
  | DataPart
  | { readonly kind: string; readonly [key: string]: unknown };

/** A single message exchanged between client and agent or between agents */
export interface A2AMessage {
  readonly role: Role;
  readonly parts: readonly Part[];
  readonly messageId?: string;
  readonly timestamp?: string;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Helper to create a simple text A2AMessage */
export function createTextMessage(
  text: string,
  role: Role = 'user',
  metadata?: Readonly<Record<string, unknown>>,
): A2AMessage {
  return {
    role,
    parts: [{ kind: 'text', text }],
    timestamp: new Date().toISOString(),
    ...(metadata ? { metadata } : {}),
  };
}

/** An artifact produced by an agent during task execution */
export interface A2AArtifact {
  readonly artifactId: string;
  readonly name?: string;
  readonly description?: string;
  readonly mimeType?: string;
  readonly uri?: string;
  readonly parts?: readonly Part[];
  readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Represents an A2A unit of work and its complete lifecycle */
export interface A2ATask {
  readonly id: string;
  readonly contextId?: string;
  readonly status: TaskStatus;
  readonly history?: readonly A2AMessage[];
  readonly artifacts?: readonly A2AArtifact[];
  readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Skill advertised by an agent in its Agent Card */
export interface AgentSkill {
  readonly id: string;
  readonly name?: string;
  readonly description: string;
  readonly tags?: readonly string[];
  readonly examples?: readonly string[];
  readonly input_modes?: readonly string[];
  readonly output_modes?: readonly string[];
  readonly security?: readonly Readonly<Record<string, readonly string[]>>[];
}

/** Capabilities flags for an agent */
export interface AgentCapabilities {
  readonly streaming?: boolean;
  readonly push_notifications?: boolean;
  readonly batch?: boolean;
  readonly state_transition_history?: boolean;
}

/** Interface / endpoint supported by an agent */
export interface AgentInterface {
  readonly url: string;
  readonly protocol_version: string;
  readonly protocol_binding?: string; // 'JSONRPC' | 'HTTP'
  readonly tenant?: string;
}

/** Security scheme declaration on an Agent Card */
export interface SecurityScheme {
  readonly type: 'http' | 'oauth2' | 'openIdConnect' | 'apiKey';
  readonly scheme?: string; // e.g. 'bearer'
  readonly bearerFormat?: string; // e.g. 'JWT'
  readonly description?: string;
  readonly openIdConnectUrl?: string;
  readonly flows?: Readonly<Record<string, unknown>>;
}

/** Security requirement for accessing an agent or skill */
export type SecurityRequirement = Readonly<Record<string, readonly string[]>>;

/** Agent Card (published at /.well-known/agent-card.json) */
export interface AgentCard {
  readonly name: string;
  readonly description?: string;
  readonly version?: string;
  readonly supported_interfaces: readonly AgentInterface[];
  readonly skills: readonly AgentSkill[];
  readonly capabilities?: AgentCapabilities;
  readonly default_input_modes?: readonly string[];
  readonly default_output_modes?: readonly string[];
  readonly security_schemes?: Readonly<Record<string, SecurityScheme>>;
  readonly security_requirements?: readonly SecurityRequirement[];
  readonly provider?: {
    readonly organization?: string;
    readonly url?: string;
  };
  readonly documentation_url?: string;
  readonly icon_url?: string;
}

/** JSON-RPC 2.0 Request shape */
export interface A2AJsonRpcRequest<TMethod extends string = string, TParams = unknown> {
  readonly jsonrpc: '2.0';
  readonly id?: string | number | null;
  readonly method: TMethod;
  readonly params?: TParams;
}

/** JSON-RPC 2.0 Error shape */
export interface A2AJsonRpcError {
  readonly code: number;
  readonly message: string;
  readonly data?: unknown;
}

/** Standard JSON-RPC 2.0 error codes */
export const JsonRpcErrorCodes = {
  PARSE_ERROR: -32700,
  INVALID_REQUEST: -32600,
  METHOD_NOT_FOUND: -32601,
  INVALID_PARAMS: -32602,
  INTERNAL_ERROR: -32603,
  UNAUTHORIZED: -32001,
  TASK_NOT_FOUND: -32004,
  TASK_TERMINATED: -32005,
} as const;

/** JSON-RPC 2.0 Response shape */
export interface A2AJsonRpcResponse<TResult = unknown> {
  readonly jsonrpc: '2.0';
  readonly id?: string | number | null;
  readonly result?: TResult;
  readonly error?: A2AJsonRpcError;
}

/** Params for SendMessage */
export interface SendMessageParams {
  readonly message: A2AMessage;
  readonly skill?: string;
  readonly taskId?: string;
  readonly contextId?: string;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Result for SendMessage */
export interface SendMessageResult {
  readonly task?: A2ATask;
  readonly message?: A2AMessage;
}

/** Params for GetTask */
export interface GetTaskParams {
  readonly taskId: string;
  readonly history?: boolean;
}

/** Result for GetTask */
export type GetTaskResult = A2ATask;

/** Params for ListTasks */
export interface ListTasksParams {
  readonly contextId?: string;
  readonly state?: TaskState;
  readonly limit?: number;
  readonly cursor?: string;
}

/** Result for ListTasks */
export interface ListTasksResult {
  readonly tasks: readonly A2ATask[];
  readonly nextCursor?: string;
}

/** Params for CancelTask */
export interface CancelTaskParams {
  readonly taskId: string;
  readonly reason?: string;
}

/** Result for CancelTask */
export interface CancelTaskResult {
  readonly task: A2ATask;
}
