import type { ChatClient } from '../chat/client/default-chat-client.ts';
import type { ChatOptions } from '../chat/prompt/chat-options.ts';
import { AiError } from '../model/errors.ts';
import type { ToolCallback } from '../tool/tool-callback.ts';
import { throwIfAborted, type WorkflowCallOptions } from './workflow-utils.ts';

/** Built-in start node id. */
export const GRAPH_START = '__start__' as const;
/** Built-in finish node id. */
export const GRAPH_FINISH = '__finish__' as const;

export type GraphNodeId = string;

/**
 * Mutable bag carried across nodes for custom state (not the typed edge value).
 */
export type GraphState = Record<string, unknown>;

/**
 * Per-node / edge execution context.
 * Inject ChatClient, tools, and arbitrary bags the same way agents do.
 */
export interface GraphNodeContext {
  readonly graphName: string;
  readonly nodeId: GraphNodeId;
  readonly step: number;
  readonly signal?: AbortSignal;
  readonly chatClient?: ChatClient;
  readonly tools?: readonly ToolCallback[];
  readonly options?: ChatOptions;
  /** Shared mutable state for the run. */
  readonly state: GraphState;
  /** Opaque bag from {@link GraphRunOptions.context}. */
  readonly context: Readonly<Record<string, unknown>>;
}

export type GraphNodeHandler<TIn = unknown, TOut = unknown> = (
  input: TIn,
  ctx: GraphNodeContext,
) => TOut | Promise<TOut>;

export type GraphEdgePredicate<T = unknown> = (
  value: T,
  ctx: GraphNodeContext,
) => boolean | Promise<boolean>;

export type GraphEdgeTransform<TIn = unknown, TOut = unknown> = (
  value: TIn,
  ctx: GraphNodeContext,
) => TOut | Promise<TOut>;

export interface GraphEdgeOptions<TIn = unknown, TOut = unknown> {
  /**
   * When set, the edge is taken only if the predicate returns true.
   * First matching edge wins (declaration order).
   */
  readonly when?: GraphEdgePredicate<TIn>;
  /** Optional transform applied to the node output before the next node. */
  readonly transform?: GraphEdgeTransform<TIn, TOut>;
  /** Optional edge name for path metadata. */
  readonly name?: string;
}

export interface GraphStepRecord {
  readonly step: number;
  readonly nodeId: GraphNodeId;
  readonly kind: 'start' | 'node' | 'subgraph' | 'finish';
  readonly input: unknown;
  readonly output: unknown;
  readonly durationMs: number;
  readonly edgeName?: string;
  readonly fromNodeId?: GraphNodeId;
}

export interface GraphRunResult<TOut = unknown> {
  readonly output: TOut;
  readonly steps: readonly GraphStepRecord[];
  /** Ordered node ids visited (including start/finish). */
  readonly path: readonly GraphNodeId[];
  readonly stepCount: number;
}

export interface GraphLifecycleHooks {
  onGraphStart?(event: { graphName: string; input: unknown }): void | Promise<void>;
  onGraphComplete?(event: {
    graphName: string;
    output: unknown;
    steps: readonly GraphStepRecord[];
  }): void | Promise<void>;
  onGraphFail?(event: {
    graphName: string;
    error: unknown;
    steps: readonly GraphStepRecord[];
    nodeId?: GraphNodeId;
  }): void | Promise<void>;
  onNodeStart?(event: {
    graphName: string;
    nodeId: GraphNodeId;
    step: number;
    input: unknown;
  }): void | Promise<void>;
  onNodeComplete?(event: {
    graphName: string;
    nodeId: GraphNodeId;
    step: number;
    input: unknown;
    output: unknown;
    durationMs: number;
  }): void | Promise<void>;
  onNodeFail?(event: {
    graphName: string;
    nodeId: GraphNodeId;
    step: number;
    input: unknown;
    error: unknown;
    durationMs: number;
  }): void | Promise<void>;
  onSubgraphStart?(event: {
    graphName: string;
    nodeId: GraphNodeId;
    subgraphName: string;
    step: number;
    input: unknown;
  }): void | Promise<void>;
  onSubgraphComplete?(event: {
    graphName: string;
    nodeId: GraphNodeId;
    subgraphName: string;
    step: number;
    output: unknown;
    durationMs: number;
  }): void | Promise<void>;
}

export interface GraphRunOptions extends WorkflowCallOptions {
  /** Hard cap on node executions (including start/finish). Default 100. */
  readonly maxSteps?: number;
  /** Shared mutable state seed. */
  readonly state?: GraphState;
  /** Opaque context bag passed to every node. */
  readonly context?: Readonly<Record<string, unknown>>;
  readonly chatClient?: ChatClient;
  readonly tools?: readonly ToolCallback[];
  readonly hooks?: GraphLifecycleHooks;
}

export interface GraphWorkflowOptions {
  readonly name?: string;
  readonly maxSteps?: number;
  readonly chatClient?: ChatClient;
  readonly tools?: readonly ToolCallback[];
  readonly hooks?: GraphLifecycleHooks;
}

// --- Internal model ---

interface GraphEdgeDef {
  readonly from: GraphNodeId;
  readonly to: GraphNodeId;
  readonly when?: GraphEdgePredicate;
  readonly transform?: GraphEdgeTransform;
  readonly name?: string;
}

type NodeKind = 'start' | 'finish' | 'action' | 'subgraph';

interface GraphNodeDef {
  readonly id: GraphNodeId;
  readonly kind: NodeKind;
  readonly handler?: GraphNodeHandler;
  readonly subgraph?: GraphWorkflow<unknown, unknown>;
}

const DEFAULT_MAX_STEPS = 100;

function graphError(
  message: string,
  code: AiError['code'] = 'invalid-request',
  details?: ConstructorParameters<typeof AiError>[2],
): AiError {
  return new AiError(message, code, { retryable: false, ...details });
}

/**
 * Validated, immutable agent graph: start → nodes → finish with typed data on edges.
 *
 * Supports linear chains, conditional branches, loops, transforming edges, and nested subgraphs.
 * Prefer fixed workflows ({@link ChainWorkflow}, etc.) when the path is known; use graphs for
 * arbitrary control flow.
 *
 * @example
 * ```ts
 * const graph = GraphWorkflow.builder<string, string>('double')
 *   .node('upper', (s) => s.toUpperCase())
 *   .edge(GRAPH_START, 'upper')
 *   .edge('upper', GRAPH_FINISH)
 *   .build();
 * const { output } = await graph.run('hi');
 * ```
 */
export class GraphWorkflow<TIn = unknown, TOut = unknown> {
  private readonly name: string;
  private readonly nodes: ReadonlyMap<GraphNodeId, GraphNodeDef>;
  private readonly edgesByFrom: ReadonlyMap<GraphNodeId, readonly GraphEdgeDef[]>;
  private readonly defaultMaxSteps: number;
  private readonly defaultChatClient?: ChatClient;
  private readonly defaultTools?: readonly ToolCallback[];
  private readonly defaultHooks?: GraphLifecycleHooks;

  /** @internal Use {@link GraphWorkflow.builder} / {@link GraphWorkflow.of}. */
  constructor(
    name: string,
    nodes: Map<GraphNodeId, GraphNodeDef>,
    edges: readonly GraphEdgeDef[],
    options: GraphWorkflowOptions,
  ) {
    this.name = name;
    this.nodes = nodes;
    const byFrom = new Map<GraphNodeId, GraphEdgeDef[]>();
    for (const e of edges) {
      const list = byFrom.get(e.from) ?? [];
      list.push(e);
      byFrom.set(e.from, list);
    }
    this.edgesByFrom = byFrom;
    this.defaultMaxSteps = options.maxSteps ?? DEFAULT_MAX_STEPS;
    this.defaultChatClient = options.chatClient;
    this.defaultTools = options.tools;
    this.defaultHooks = options.hooks;
  }

  get graphName(): string {
    return this.name;
  }

  /** Start a fluent builder. */
  static builder<TIn = unknown, TOut = unknown>(
    name = 'graph',
    options?: GraphWorkflowOptions,
  ): GraphWorkflowBuilder<TIn, TOut> {
    return new GraphWorkflowBuilder<TIn, TOut>(name, options ?? {});
  }

  static of<TIn = unknown, TOut = unknown>(
    name: string,
    configure: (b: GraphWorkflowBuilder<TIn, TOut>) => void,
    options?: GraphWorkflowOptions,
  ): GraphWorkflow<TIn, TOut> {
    const builder = GraphWorkflow.builder<TIn, TOut>(name, options);
    configure(builder);
    return builder.build();
  }

  /**
   * Validate structure without running. Throws {@link AiError} when invalid.
   * {@link build} already validates; call again after composition if needed.
   */
  validate(): void {
    validateGraph(this.name, this.nodes, this.edgesByFrom);
  }

  async run(input: TIn, options?: GraphRunOptions): Promise<GraphRunResult<TOut>> {
    throwIfAborted(options?.signal);
    const hooks = mergeHooks(this.defaultHooks, options?.hooks);
    const maxSteps = options?.maxSteps ?? this.defaultMaxSteps;
    const state: GraphState = { ...(options?.state ?? {}) };
    const bag = options?.context ?? {};
    const chatClient = options?.chatClient ?? this.defaultChatClient;
    const tools = options?.tools ?? this.defaultTools;
    const steps: GraphStepRecord[] = [];
    const path: GraphNodeId[] = [];

    await hooks.onGraphStart?.({ graphName: this.name, input });

    let currentNodeId: GraphNodeId = GRAPH_START;
    let currentValue: unknown = input;
    let step = 0;
    let lastEdgeName: string | undefined;
    let lastFrom: GraphNodeId | undefined;

    try {
      while (true) {
        throwIfAborted(options?.signal);
        step += 1;
        if (step > maxSteps) {
          throw graphError(
            `Graph "${this.name}" exceeded maxSteps (${maxSteps}) at node "${currentNodeId}"`,
          );
        }

        const node = this.nodes.get(currentNodeId);
        if (!node) {
          throw graphError(`Graph "${this.name}" has no node "${currentNodeId}"`);
        }

        const ctx: GraphNodeContext = {
          graphName: this.name,
          nodeId: currentNodeId,
          step,
          signal: options?.signal,
          chatClient,
          tools,
          options: options?.options,
          state,
          context: bag,
        };

        const started = Date.now();
        await hooks.onNodeStart?.({
          graphName: this.name,
          nodeId: currentNodeId,
          step,
          input: currentValue,
        });

        let output: unknown;
        try {
          if (node.kind === 'start') {
            output = currentValue;
          } else if (node.kind === 'finish') {
            output = currentValue;
            const durationMs = Date.now() - started;
            const record: GraphStepRecord = {
              step,
              nodeId: currentNodeId,
              kind: 'finish',
              input: currentValue,
              output,
              durationMs,
              edgeName: lastEdgeName,
              fromNodeId: lastFrom,
            };
            steps.push(record);
            path.push(currentNodeId);
            await hooks.onNodeComplete?.({
              graphName: this.name,
              nodeId: currentNodeId,
              step,
              input: currentValue,
              output,
              durationMs,
            });
            const result: GraphRunResult<TOut> = {
              output: output as TOut,
              steps,
              path,
              stepCount: step,
            };
            await hooks.onGraphComplete?.({
              graphName: this.name,
              output,
              steps,
            });
            return result;
          } else if (node.kind === 'subgraph' && node.subgraph) {
            await hooks.onSubgraphStart?.({
              graphName: this.name,
              nodeId: currentNodeId,
              subgraphName: node.subgraph.graphName,
              step,
              input: currentValue,
            });
            const sub = await node.subgraph.run(currentValue, {
              signal: options?.signal,
              maxSteps: options?.maxSteps,
              state,
              context: bag,
              chatClient,
              tools,
              options: options?.options,
              hooks,
            });
            output = sub.output;
            // Flatten subgraph path into parent path for debugging.
            for (const p of sub.path) {
              path.push(`${currentNodeId}/${p}`);
            }
            await hooks.onSubgraphComplete?.({
              graphName: this.name,
              nodeId: currentNodeId,
              subgraphName: node.subgraph.graphName,
              step,
              output,
              durationMs: Date.now() - started,
            });
          } else if (node.handler) {
            output = await node.handler(currentValue, ctx);
          } else {
            throw graphError(`Graph "${this.name}" node "${currentNodeId}" has no handler`);
          }

          const durationMs = Date.now() - started;
          const kind: GraphStepRecord['kind'] =
            node.kind === 'subgraph' ? 'subgraph' : node.kind === 'start' ? 'start' : 'node';
          steps.push({
            step,
            nodeId: currentNodeId,
            kind,
            input: currentValue,
            output,
            durationMs,
            edgeName: lastEdgeName,
            fromNodeId: lastFrom,
          });
          if (node.kind !== 'subgraph') {
            path.push(currentNodeId);
          }
          await hooks.onNodeComplete?.({
            graphName: this.name,
            nodeId: currentNodeId,
            step,
            input: currentValue,
            output,
            durationMs,
          });
        } catch (error) {
          const durationMs = Date.now() - started;
          await hooks.onNodeFail?.({
            graphName: this.name,
            nodeId: currentNodeId,
            step,
            input: currentValue,
            error,
            durationMs,
          });
          throw error;
        }

        // Select next edge
        const outgoing = this.edgesByFrom.get(currentNodeId) ?? [];
        if (outgoing.length === 0) {
          throw graphError(
            `Graph "${this.name}": no outgoing edges from "${currentNodeId}" (and not finish)`,
          );
        }

        let chosen: GraphEdgeDef | undefined;
        for (const edge of outgoing) {
          if (!edge.when) {
            chosen = edge;
            break;
          }
          const edgeCtx: GraphNodeContext = { ...ctx, nodeId: currentNodeId };
          const ok = await edge.when(output, edgeCtx);
          if (ok) {
            chosen = edge;
            break;
          }
        }

        if (!chosen) {
          throw graphError(
            `Graph "${this.name}": no matching edge from "${currentNodeId}" after step ${step}`,
          );
        }

        let nextValue = output;
        if (chosen.transform) {
          nextValue = await chosen.transform(output, ctx);
        }

        lastEdgeName = chosen.name;
        lastFrom = currentNodeId;
        currentNodeId = chosen.to;
        currentValue = nextValue;
      }
    } catch (error) {
      await hooks.onGraphFail?.({
        graphName: this.name,
        error,
        steps,
        nodeId: currentNodeId,
      });
      throw error;
    }
  }
}

/**
 * Fluent builder for {@link GraphWorkflow}.
 * Call {@link build} once; the resulting graph is immutable and validated.
 */
export class GraphWorkflowBuilder<TIn = unknown, TOut = unknown> {
  private readonly name: string;
  private readonly options: GraphWorkflowOptions;
  private readonly nodes = new Map<GraphNodeId, GraphNodeDef>();
  private readonly edges: GraphEdgeDef[] = [];
  private sealed = false;

  constructor(name: string, options: GraphWorkflowOptions) {
    this.name = name;
    this.options = options;
    this.nodes.set(GRAPH_START, { id: GRAPH_START, kind: 'start' });
    this.nodes.set(GRAPH_FINISH, { id: GRAPH_FINISH, kind: 'finish' });
  }

  /** Register an action node. */
  node<TNodeIn = unknown, TNodeOut = unknown>(
    id: GraphNodeId,
    handler: GraphNodeHandler<TNodeIn, TNodeOut>,
  ): this {
    this.assertOpen();
    this.assertUserId(id);
    if (this.nodes.has(id)) {
      throw graphError(`Graph "${this.name}": duplicate node id "${id}"`);
    }
    this.nodes.set(id, {
      id,
      kind: 'action',
      handler: handler as GraphNodeHandler,
    });
    return this;
  }

  /**
   * Nest another graph as a single node (typed in/out of the subgraph).
   */
  subgraph<TSubIn = unknown, TSubOut = unknown>(
    id: GraphNodeId,
    child: GraphWorkflow<TSubIn, TSubOut>,
  ): this {
    this.assertOpen();
    this.assertUserId(id);
    if (this.nodes.has(id)) {
      throw graphError(`Graph "${this.name}": duplicate node id "${id}"`);
    }
    this.nodes.set(id, {
      id,
      kind: 'subgraph',
      subgraph: child as GraphWorkflow<unknown, unknown>,
    });
    return this;
  }

  /**
   * Direct or conditional / transforming edge.
   * Use {@link GRAPH_START} and {@link GRAPH_FINISH} for terminals.
   */
  edge<TEdgeIn = unknown, TEdgeOut = unknown>(
    from: GraphNodeId,
    to: GraphNodeId,
    options?: GraphEdgeOptions<TEdgeIn, TEdgeOut>,
  ): this {
    this.assertOpen();
    this.edges.push({
      from,
      to,
      when: options?.when as GraphEdgePredicate | undefined,
      transform: options?.transform as GraphEdgeTransform | undefined,
      name: options?.name,
    });
    return this;
  }

  /** Convenience: unconditional edge. */
  link(from: GraphNodeId, to: GraphNodeId, name?: string): this {
    return this.edge(from, to, name ? { name } : undefined);
  }

  /** Attach default ChatClient / tools / hooks / maxSteps for runs. */
  withDefaults(options: GraphWorkflowOptions): this {
    this.assertOpen();
    Object.assign(this.options, options);
    return this;
  }

  build(): GraphWorkflow<TIn, TOut> {
    this.assertOpen();
    validateGraph(this.name, this.nodes, groupEdges(this.edges));
    this.sealed = true;
    return new GraphWorkflow<TIn, TOut>(this.name, new Map(this.nodes), [...this.edges], {
      ...this.options,
    });
  }

  private assertOpen(): void {
    if (this.sealed) {
      throw graphError(`Graph builder "${this.name}" is already built`);
    }
  }

  private assertUserId(id: GraphNodeId): void {
    if (id === GRAPH_START || id === GRAPH_FINISH) {
      throw graphError(`Graph "${this.name}": "${id}" is reserved`);
    }
    if (!id || id.includes('/')) {
      throw graphError(`Graph "${this.name}": invalid node id "${id}"`);
    }
  }
}

function groupEdges(
  edges: readonly GraphEdgeDef[],
): ReadonlyMap<GraphNodeId, readonly GraphEdgeDef[]> {
  const byFrom = new Map<GraphNodeId, GraphEdgeDef[]>();
  for (const e of edges) {
    const list = byFrom.get(e.from) ?? [];
    list.push(e);
    byFrom.set(e.from, list);
  }
  return byFrom;
}

function validateGraph(
  name: string,
  nodes: ReadonlyMap<GraphNodeId, GraphNodeDef>,
  edgesByFrom: ReadonlyMap<GraphNodeId, readonly GraphEdgeDef[]>,
): void {
  if (!nodes.has(GRAPH_START) || !nodes.has(GRAPH_FINISH)) {
    throw graphError(`Graph "${name}" must include start and finish nodes`);
  }

  const allEdges: GraphEdgeDef[] = [];
  for (const list of edgesByFrom.values()) {
    allEdges.push(...list);
  }

  // Finish must not have outgoing edges.
  if ((edgesByFrom.get(GRAPH_FINISH) ?? []).length > 0) {
    throw graphError(`Graph "${name}": finish node cannot have outgoing edges`);
  }

  // Valid targets and sources.
  for (const edge of allEdges) {
    if (!nodes.has(edge.from)) {
      throw graphError(`Graph "${name}": edge from unknown node "${edge.from}"`);
    }
    if (!nodes.has(edge.to)) {
      throw graphError(`Graph "${name}": edge to unknown node "${edge.to}"`);
    }
    if (edge.from === GRAPH_FINISH) {
      throw graphError(`Graph "${name}": edge from finish is not allowed`);
    }
  }

  // Start must have at least one outgoing edge.
  if ((edgesByFrom.get(GRAPH_START) ?? []).length === 0) {
    throw graphError(`Graph "${name}": start has no outgoing edges`);
  }

  // Every non-finish node that is reachable should be able to leave
  // (except we allow unreachable nodes only as a soft warning — reject orphans that are
  // referenced but have no exit if they are action nodes with zero outs).
  for (const [id, node] of nodes) {
    if (node.kind === 'finish') continue;
    const outs = edgesByFrom.get(id) ?? [];
    if (outs.length === 0 && id !== GRAPH_FINISH) {
      // Only enforce for start and nodes that are edge targets / sources.
      const isReferenced = id === GRAPH_START || allEdges.some((e) => e.from === id || e.to === id);
      if (isReferenced) {
        throw graphError(
          `Graph "${name}": node "${id}" has no outgoing edges (must reach finish or loop)`,
        );
      }
    }
  }

  // Reachability: finish must be reachable from start (ignore predicates — structural).
  const reachable = new Set<GraphNodeId>();
  const queue: GraphNodeId[] = [GRAPH_START];
  while (queue.length) {
    const id = queue.shift();
    if (id === undefined) break;
    if (reachable.has(id)) continue;
    reachable.add(id);
    for (const e of edgesByFrom.get(id) ?? []) {
      if (!reachable.has(e.to)) queue.push(e.to);
    }
  }
  if (!reachable.has(GRAPH_FINISH)) {
    throw graphError(`Graph "${name}": finish is not reachable from start`);
  }

  // Every edge endpoint should be reachable from start (no dead fragments that look wired).
  for (const edge of allEdges) {
    if (!reachable.has(edge.from)) {
      throw graphError(`Graph "${name}": edge from unreachable node "${edge.from}"`);
    }
  }
}

function mergeHooks(base?: GraphLifecycleHooks, extra?: GraphLifecycleHooks): GraphLifecycleHooks {
  if (!base) return extra ?? {};
  if (!extra) return base;
  return {
    onGraphStart: chainHook(base.onGraphStart, extra.onGraphStart),
    onGraphComplete: chainHook(base.onGraphComplete, extra.onGraphComplete),
    onGraphFail: chainHook(base.onGraphFail, extra.onGraphFail),
    onNodeStart: chainHook(base.onNodeStart, extra.onNodeStart),
    onNodeComplete: chainHook(base.onNodeComplete, extra.onNodeComplete),
    onNodeFail: chainHook(base.onNodeFail, extra.onNodeFail),
    onSubgraphStart: chainHook(base.onSubgraphStart, extra.onSubgraphStart),
    onSubgraphComplete: chainHook(base.onSubgraphComplete, extra.onSubgraphComplete),
  };
}

function chainHook<E>(
  a?: (event: E) => void | Promise<void>,
  b?: (event: E) => void | Promise<void>,
): ((event: E) => Promise<void>) | undefined {
  if (!a && !b) return undefined;
  return async (event: E) => {
    if (a) await a(event);
    if (b) await b(event);
  };
}

export function graphWorkflow<TIn = unknown, TOut = unknown>(
  name?: string,
  options?: GraphWorkflowOptions,
): GraphWorkflowBuilder<TIn, TOut> {
  return GraphWorkflow.builder<TIn, TOut>(name, options);
}

// ---------------------------------------------------------------------------
// Built-in: simple LLM + tool loop (uses ChatClient tool-calling advisor)
// ---------------------------------------------------------------------------

export interface ChatToolLoopGraphOptions extends GraphWorkflowOptions {
  readonly system?: string;
  /**
   * Max model calls inside the loop node. Default 8.
   * (Outer graph maxSteps still applies.)
   */
  readonly maxToolRounds?: number;
}

export interface ChatToolLoopInput {
  readonly message: string;
  readonly system?: string;
}

export interface ChatToolLoopOutput {
  readonly content: string;
}

/**
 * Single-node graph that runs a ChatClient prompt with tools (ToolCallingAdvisor loop).
 * Useful as a building block or nested subgraph.
 */
export function chatToolLoopGraph(
  options: ChatToolLoopGraphOptions & { chatClient: ChatClient },
): GraphWorkflow<ChatToolLoopInput | string, ChatToolLoopOutput> {
  const maxToolRounds = options.maxToolRounds ?? 8;
  return GraphWorkflow.builder<ChatToolLoopInput | string, ChatToolLoopOutput>(
    options.name ?? 'chat-tool-loop',
    options,
  )
    .node('chat', async (input: ChatToolLoopInput | string, ctx) => {
      const client = ctx.chatClient ?? options.chatClient;
      if (!client) {
        throw graphError('chatToolLoopGraph requires a ChatClient');
      }
      throwIfAborted(ctx.signal);
      const message = typeof input === 'string' ? input : input.message;
      const system = (typeof input === 'string' ? undefined : input.system) ?? options.system;

      let spec = client.prompt();
      if (system) spec = spec.system(system);
      spec = spec.user(message);
      const tools = ctx.tools ?? options.tools;
      if (tools?.length) {
        spec = spec.tools(...tools);
      }
      // maxToolRounds reserved for future multi-node tool loop strategy.
      void maxToolRounds;
      const merged: ChatOptions = {
        ...ctx.options,
        signal: ctx.signal ?? ctx.options?.signal,
      };
      if (merged.signal !== undefined || ctx.options) {
        spec = spec.options(merged);
      }
      const content = (await spec.call().content()) ?? '';
      return { content };
    })
    .edge(GRAPH_START, 'chat')
    .edge('chat', GRAPH_FINISH)
    .build();
}

/**
 * Multi-node graph strategy: model node → conditional finish vs continue.
 * Relies on ChatClient's built-in tool-calling so tools run inside the model call;
 * the graph records each model round as a separate step when you fan out manually.
 *
 * This helper keeps a single chat node for the common case (advisor loop).
 * For explicit model/tools nodes, compose with {@link GraphWorkflow.builder}.
 */
export function simpleAgentGraph(
  options: ChatToolLoopGraphOptions & { chatClient: ChatClient },
): GraphWorkflow<string, string> {
  const loop = chatToolLoopGraph(options);
  return GraphWorkflow.builder<string, string>(options.name ?? 'simple-agent', options)
    .subgraph('agent', loop)
    .edge(GRAPH_START, 'agent', {
      transform: (message: string) => ({ message, system: options.system }),
    })
    .edge('agent', GRAPH_FINISH, {
      transform: (out: ChatToolLoopOutput) => out.content,
    })
    .build();
}
