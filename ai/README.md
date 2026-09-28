# @di-framework/ai

Spring AI–aligned chat, tools, RAG, MCP, and agents for TypeScript. Portable model abstractions sit on top of OpenAI-compatible and Anthropic HTTP adapters (no vendor SDKs). Wire everything into `@di-framework/core` with annotations (`@AiService`, `@Agent`, `@Tool`, …) or `configureAi`.

**Style:** prefer `static of` / `static builder` factories and free functions for pure helpers; keep instance methods for stateful clients and fluent builders. See [docs/static-methods-convention.md](../../docs/static-methods-convention.md).

## Features

- **Annotation DX**: `@AiService` / `@Agent` assistants, `@Tool` / `@ToolSet` beans, `@WithMemory` / `@WithRag` / `@WithTools`, workflows (`@Chain`, `@Route`, …).
- **ChatClient**: fluent prompt / call / stream API with an advisor chain (memory, tools, RAG, logging, observation).
- **Prototype builder**: inject `AiTokens.CHAT_CLIENT_BUILDER` (fresh per resolve) like Spring’s `ChatClient.Builder`.
- **Providers**: `OpenAiChatModel` and `AnthropicChatModel` over `fetch` — no official SDKs.
- **Tools**: `functionToolCallback`, method-level `@Tool` on DI beans, automatic tool-calling loops.
- **Structured output**: JSON Schema converters and `call().entity(...)`.
- **Memory / RAG / MCP / agents**: same runtime as before, now annotation-friendly.

## Installation

```bash
bun add @di-framework/ai @di-framework/core
# or
npm install @di-framework/ai @di-framework/core
```

Peer: `@di-framework/core`. Runtime dependency: `@modelcontextprotocol/sdk` (MCP helpers). Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` when using the HTTP providers.

## Annotation-first quick start

```ts
import { Container } from '@di-framework/core/decorators';
import {
  Agent,
  AiService,
  configureAi,
  OpenAiChatModel,
  resolveAiService,
  resolveAnnotatedAgent,
  SystemMessageAnn,
  Tool,
  ToolParam,
  ToolSet,
  UserMessageAnn,
  WithMemory,
  MemoryId,
} from '@di-framework/ai';

@ToolSet()
@Container()
class WeatherTools {
  @Tool({
    description: 'Get weather for a city',
    inputSchema: {
      type: 'object',
      properties: { city: { type: 'string' } },
      required: ['city'],
    },
  })
  getWeather(@ToolParam('City name') input: { city: string }) {
    return { temp: 68, city: input.city };
  }
}

@AiService({ tools: [WeatherTools] })
@WithMemory()
class WeatherBot {
  @SystemMessageAnn('You help with weather questions.')
  ask(@UserMessageAnn() question: string, @MemoryId() sessionId: string): Promise<string> {
    throw new Error('handled by AiService proxy');
  }
}

@Agent({
  system: 'You help with weather.',
  tools: [WeatherTools],
})
class WeatherAgent {}

configureAi({
  chatModel: new OpenAiChatModel({ model: 'gpt-4o-mini' }),
  toolBeans: [WeatherTools],
  memory: /* MessageWindowChatMemory… */ undefined,
});

const bot = resolveAiService(WeatherBot);
await bot.ask('Weather in Yorktown?', 'session-1');

const agent = resolveAnnotatedAgent(WeatherAgent);
await agent.chat('Weather in Yorktown?');
```

Parameter decorators are factories — use `@UserMessageAnn()`, `@MemoryId()`, `@ToolParam()` (with parentheses).

### Inject a prototype `ChatClient.Builder`

```ts
import { Component, Container } from '@di-framework/core/decorators';
import {
  AiTokens,
  ChatAgent,
  type ChatClientBuilder,
  configureAi,
  OpenAiChatModel,
} from '@di-framework/ai';

configureAi({ chatModel: new OpenAiChatModel() });

@Container()
class SupportAgentService {
  private readonly agent: ChatAgent;

  constructor(@Component(AiTokens.CHAT_CLIENT_BUILDER) builder: ChatClientBuilder) {
    this.agent = ChatAgent.fromBuilder(builder)
      .system('You are an e-commerce support assistant.')
      .build();
  }

  chat(prompt: string, sessionId: string) {
    return this.agent.chat(prompt, { conversationId: sessionId });
  }
}
```

### Decorator catalog (selection)

| Decorator | Purpose |
| --- | --- |
| `@AiService` / `@Assistant` | Declarative chat assistant (class → proxy) |
| `@Agent` / `@ChatAgentBean` | Declarative `ChatAgent` bean |
| `@SystemMessageAnn` / `@UserMessageAnn()` / `@MemoryId()` | Prompt + session wiring |
| `@Tool` / `@ToolSet` / `@ToolParam()` | Tool methods on beans |
| `@WithMemory` / `@WithRag` / `@WithTools` / `@AiObserved` | Attach advisors |
| `@EnableAi` | Bootstrap scanning + `configureAi` options on an app class |
| `@Chain` / `@Route` / `@Parallel` / … | Workflow stereotypes |

Where names collide with runtime types, the package exports `AiAdvisor`, `ChatModelAnn`, `SystemMessageAnn`, `UserMessageAnn`, `AssistantMessageAnn`, `VectorStoreAnn`, `ChatMemoryAnn`, `DocumentAnn`, `EmbeddingModelAnn`, `ChatClientAnn`, and `PromptTemplate` (prompt decorator; `Prompt` remains the message class).

## Imperative ChatClient

```ts
import { ChatClient, OpenAiChatModel } from '@di-framework/ai';

const model = new OpenAiChatModel({ model: 'gpt-4o-mini' });
const client = ChatClient.create(model);

const answer = await client
  .prompt()
  .system('You are concise.')
  .user('What is Yorktown known for?')
  .call()
  .content();
```

## DI with `configureAi`

```ts
configureAi({
  chatModel: new OpenAiChatModel(),
  defaultSystem: 'You help with weather questions.',
  toolBeans: [WeatherTools],
  observation: true,
  agent: true, // optional AiTokens.CHAT_AGENT
  scanAnnotations: true, // default — processes @AiService / @Agent / …
});
```

## Tools, memory, RAG, MCP, workflows

Imperative APIs are unchanged: `functionToolCallback`, `MessageChatMemoryAdvisor`, `RetrievalAugmentationAdvisor`, MCP adapters, and `ChainWorkflow` / `RoutingWorkflow` / … See source tests under `tests/` for examples.

### Explicit context compression

`ContextCompressionAdvisor` runs after chat-memory loading and before skill
retrieval on both call and stream paths. It requires an application-supplied
`TokenCounter` and `ContextCompressor`; the package does not estimate tokens or
invoke a hidden model.

```typescript
const compression = new ContextCompressionAdvisor({
  tokenBudget: 8_000,
  tokenCounter: myTokenizer,
  compressor: mySummarizer,
  persistence: 'request', // default; use "memory" only with ReplaceableChatMemory
});
```

Compressor output is rejected before model invocation if it remains over budget,
is malformed, or changes system messages, the current user turn, media, or an
assistant/tool-response group. `MessageWindowChatMemory` supports atomic
replacement for opt-in persistent compression. `onCompression` reports token
counts, compressed ranges, persistence, and duration without message bodies.

**Agent Skills** (`SKILL.md`, progressive disclosure) are not in this package. Use [`@di-framework/ai-utils`](../di-framework-ai-utils) `SkillsAgent.builder()` / `SkillsToolbox.builder()`. Docs: [Agent Skills](../../docs/Writerside/topics/ai-utils.md).

### Tool Execution Authorization & Interception

Tool execution can be intercepted and authorized by registering ordered `ToolExecutionAdvisor` instances with `ToolCallingManager` / `DefaultToolCallingManager`.

`ToolAuthorizationAdvisor` integrates with `@di-framework/auth` `AuthorizationManager` to evaluate authorization decisions before executing any tool call.

- **Covered Execution Paths**: Interception occurs inside `DefaultToolCallingManager.executeOne()` whenever the model requests a tool call (including multi-call turns).
- **Direct Execution Note**: Direct `ToolCallback.call()` invocations performed manually outside of `ToolCallingManager` are **not implicitly intercepted**; authorization policies are enforced during manager-driven execution.
- **Trusted Principal**: The authenticated subject (`Principal`) is resolved from trusted `ToolContext` data (`toolContext.get('principal')` or custom resolver) — model-generated arguments (`toolCall.arguments`) can **never** supply or overwrite the principal.
- **Fail Closed**: Missing principal, policy denial, manager exception, or unresolvable manager configurations fail closed and return generic response `"Tool execution unauthorized"` without leaking policy decision details to the model.

```ts
import {
  createToolCallingManager,
  functionToolCallback,
  ToolAuthorizationAdvisor,
  toolCallbacksFromBean,
  Tool,
  ToolSet,
} from '@di-framework/ai';
import type { AuthorizationManager, ToolAuthorizationContext } from '@di-framework/ai';

// 1. Authorization Manager definition
const authManager: AuthorizationManager<ToolAuthorizationContext> = {
  async authorize(principal, context) {
    // context.transport === 'ai-tool'
    // context.tool === tool name
    // context.arguments === parsed tool arguments
    // context.metadata === opaque auth metadata from @Tool / @ToolSet / callback
    if (principal?.sub === 'admin') return { allowed: true };
    return { allowed: false, reason: 'Insufficient privileges' };
  },
};

// 2. Opaque authorization metadata on callbacks / beans
const deleteUserTool = functionToolCallback({
  name: 'deleteUser',
  auth: { permission: 'users:delete' },
  call: ({ userId }: { userId: string }) => `Deleted ${userId}`,
});

@ToolSet({ auth: { scope: 'admin' } })
class AdminTools {
  @Tool({ auth: { permission: 'system:shutdown' } })
  shutdown() {
    return 'System shutting down';
  }
}

// 3. Create ToolCallingManager with Authorization Advisor
const toolManager = createToolCallingManager({
  authorizationManager: authManager,
});
```

### Graph workflows

For arbitrary agent control flow (branches, loops, nested subgraphs), use the typed graph runtime:

```ts
import {
  ChatClient,
  GRAPH_FINISH,
  GRAPH_START,
  GraphWorkflow,
  chatToolLoopGraph,
  functionToolCallback,
} from '@di-framework/ai';

const graph = GraphWorkflow.builder<number, string>('example')
  .node('double', (n) => n * 2)
  .node('label', (n) => `value=${n}`)
  .edge(GRAPH_START, 'double')
  .edge('double', 'label')
  .edge('label', GRAPH_FINISH)
  .build();

const { output, path, steps } = await graph.run(21, { maxSteps: 50, signal });
// output === "value=42"

// LLM + tools as a ready-made graph (uses ChatClient tool-calling)
const agentGraph = chatToolLoopGraph({
  chatClient: ChatClient.create(model),
  tools: [weatherTool],
  system: 'Help with weather.',
});
const answer = await agentGraph.run({ message: 'Weather in Yorktown?' });
```

Edges support async `when` predicates and `transform` functions. Nested graphs use `.subgraph(id, childGraph)`. Validation runs at `build()` (reachable finish, no edges from finish, unique ids). Lifecycle hooks (`onNodeStart`, `onGraphComplete`, …) align with observation-friendly debugging. Prefer fixed workflows when the path is known; use graphs when control flow must be composed dynamically. Stereotype annotations for graphs are deferred until the imperative API stabilizes.

### Planner–executor

Iterative plan → act → replan on `ChatClient` (+ tools), with `AbortSignal`, `maxSteps`, and cycle protection:

```ts
import { ChatClient, PlannerExecutorWorkflow, functionToolCallback } from '@di-framework/ai';

const pe = PlannerExecutorWorkflow.of(ChatClient.create(model));
const { answer, plan, rounds } = await pe.run('Weather in Yorktown?', {
  tools: [weatherTool],
  maxSteps: 6,
  signal,
});
```

### Network A2A 1.0 (Agent-to-Agent Protocol over HTTP)

`@di-framework/ai` implements the standard [AAIF Agent-to-Agent (A2A) 1.0 Protocol](https://github.com/google/A2A) over HTTP JSON-RPC.

- **Agent Cards**: Discover skills and capabilities at `GET {url}/.well-known/agent-card.json`.
- **JSON-RPC Operations**: Standard 1.0 methods (`SendMessage`, `GetTask`, `ListTasks`, `CancelTask`) over HTTP POST.
- **Task Lifecycle**: Strict states (`submitted` → `working` → `completed` | `failed` | `canceled` | `rejected` | `input-required` | `auth-required`).
- **Discovery**: `A2ADirectory` fetches cards from registered origins and returns connected `A2AClient` instances.
- **Process Boundary Opacity**: Wire representation carries only messages, tasks, and artifacts. Internal prompts, tool definitions (MCP), and memory keys remain completely private inside the serving process.
- **MCP vs A2A**: MCP equips agents with internal tools and resources; A2A dispatches tasks and work across independent agent services over the network.

#### Exposing an Agent over A2A

```ts
import { Agent, EnableAi } from '@di-framework/ai';

@Agent({
  name: 'ReviewAgent',
  description: 'Automated code review agent',
  skills: [{ id: 'dev.review', description: 'Review pull request diffs' }],
  a2a: { url: 'https://agents.example.com/review' },
  system: 'You are an expert code reviewer.',
  tools: [/* internal MCP tools */],
})
export class ReviewAgent {}

@EnableAi({
  a2a: true, // opts into serving Agent Card and JSON-RPC HTTP handlers
})
export class AppModule {}
```

#### Discovering and Dispatching Work via `A2ADirectory` and `A2AClient`

```ts
import { A2ADirectory, A2AClient } from '@di-framework/ai';

// Initialize directory with remote origins
const directory = A2ADirectory.create({
  origins: [
    'https://agents.example.com/aria',
    'https://agents.example.com/ravi',
  ],
});

// Discover a peer advertising the required skill
const reviewer: A2AClient = await directory.find({ skill: 'dev.review' });

// Dispatch task and await completion
const task = await reviewer.sendAndWait({
  skill: 'dev.review',
  message: 'git diff main...feature',
  metadata: { workId: 'ticket-456' },
});

console.log(task.status.state); // 'completed'
console.log(task.artifacts); // array of A2AArtifact
```

### In-process Local Bus (Non-network)

> [!NOTE]
> `A2ABus` is an in-process memory event bus for local co-located callbacks and is **not** the network A2A protocol. Use `A2ADirectory` and `A2AClient` for standard network A2A 1.0 communication.

```ts
import { A2ABus } from '@di-framework/ai';

const bus = A2ABus.create();
bus.register('researcher', async (msg) => `notes:${msg.content}`);
const reply = await bus.request('user', 'researcher', 'topic');
```

## Providers

```ts
new OpenAiChatModel({
  apiKey: process.env.OPENAI_API_KEY,
  baseUrl: 'https://api.openai.com/v1',
  model: 'gpt-4o-mini',
});

new AnthropicChatModel({
  apiKey: process.env.ANTHROPIC_API_KEY,
  model: 'claude-sonnet-4-20250514',
});
```

## Testing

```ts
import { ChatClient, ScriptedChatModel, toolCall, toolCallResponse } from '@di-framework/ai';

const model = new ScriptedChatModel([
  { respond: toolCallResponse([toolCall('c1', 'getWeather', { city: 'Yorktown' })]) },
  { respond: '68F in Yorktown' },
]);
```

## License

Licensed under either [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE), at your option.
# Durable vector stores

The package exports `BunSqliteVectorStore`, `VectorizeVectorStore`, `PgVectorStore`, and `S3VectorStore` (for AWS S3 Vectors). All implement the same `VectorStore` API and can be passed to RAG advisors. `SearchRequest.queryEmbedding` and `Document.embedding` let callers pass precomputed vectors. The Bun store persists float32 BLOBs, exact-scans small tables, and builds a portable HNSW graph for larger ones (`searchMode: 'auto' | 'exact' | 'ann'`). Optional `wasm-similarity` accelerates exact cosine ranking and is not required. Vectorize and pgvector delegate ranking to their managed backends. `S3VectorStore` supports serverless vector search and metadata filtering directly in AWS S3 Vectors with built-in AST filter translation (`translateS3FilterExpression`). Create provider schemas and indexes out of band and keep provider clients optional so Workers and Bun bundles do not pull external SDK dependencies.

## Select API or subscription access

`createChatModel()` synchronously selects a model. The caller supplies that model
through constructor injection; tests can supply `FakeChatModel` instead.

```ts
import { createChatModel, type ChatModel, Prompt } from '@di-framework/ai';

class ExampleAgent {
  constructor(private readonly model: ChatModel) {}
  async answer(text: string) {
    return (await this.model.call(new Prompt(text))).result?.output.text;
  }
}

const agent = new ExampleAgent(createChatModel({
  provider: 'openai',
  auth: 'subscription',
}));
```

For environment-driven programs, construct with `createChatModel()` and run:

```sh
PROVIDER=openai AUTH=subscription bun my-program.ts
```

Explicit `provider`, `auth`, and `model` override `PROVIDER`, `AUTH`, and
`MODEL`. A provider is required. `api.model` also precedes `MODEL`; the top-level
`model` wins over both. `env` supplies selection and API-key values and overrides
inherited environment entries for CLI children. It does not isolate the native
CLI from its saved configuration or other inherited environment variables.

| Provider | API route | Subscription route |
| --- | --- | --- |
| `openai` | OpenAI; existing default model | Codex CLI |
| `anthropic` | Anthropic; existing default model | Claude CLI |
| `xai` | xAI OpenAI-compatible endpoint; model required | Grok CLI |
| `agy` | Unsupported | AGY CLI |
| `junie` | Unsupported | Junie CLI |
| `hermes` | Unsupported | Existing Nous proxy; model required |

Vendor names default to API access. Aliases `codex`, `claude`, and `grok`, and
subscription-only providers, default to subscription access. Explicit `auth`
always wins, including on aliases. CLI subscriptions use their native model
default unless overridden. API keys come from `api.apiKey` or the corresponding
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `XAI_API_KEY`; xAI never borrows an
OpenAI key. Existing HTTP constructors keep their original behavior.

`api` accepts the existing HTTP provider options (including a custom `fetch`).
`subscription` accepts `timeoutMs` (default 120000), `maxCalls` (default 32),
`executable`, `toolCallingManager`, and `onEvent`. Irrelevant route options
are rejected. Existing DI wiring is sufficient:

```ts
configureAi({ chatModel: () => createChatModel() });
```

### Subscription setup and boundaries

CLI inference requires **Bun** and the selected CLI on PATH. Importing the package,
creating models, and calling HTTP models remain portable. Sign in using the
native subscription flow: `codex login` (ChatGPT), Claude's interactive login,
`grok login`, or AGY's Google sign-in. Junie uses its native account setup
(including `JUNIE_API_KEY` for a Junie token). Saved CLI configuration determines
the account access; the factory neither extracts credentials nor changes login
configuration.

Subscription mode rejects explicit API options and known conflicting API keys,
base URLs, or alternate provider-authentication environment settings. CLI
environments are checked again when inference starts. There is **no API
fallback** on missing executables, login errors, timeouts, or provider failures.
This validation cannot establish which account a saved CLI configuration uses.

Hermes uses `http://127.0.0.1:8645/v1` with the existing placeholder bearer.
Start `hermes proxy start --provider nous` separately and select a Portal model.
The factory does not start the proxy or read its credentials.

Native CLIs own their internal model/tool loop. Framework callbacks are exposed
through a private MCP child and authenticated loopback transport. Arguments are
schema-validated, executions are sequential, and duplicate call IDs reuse the
result. Tool context stays in the host. A manager configured on
`ToolCallingAdvisor` takes precedence over the subscription model's manager;
direct model calls use the explicit manager or the framework default. Authorization
advisors therefore remain active.

Responses contain completed text and `metadata.bridgeEvents`, with no pending
tool calls to execute again. Plain text works without MCP discovery; callbacks
require discovery. The plugin playground separately checks its promised
`Skill` invocation. Tool-result events report the manager's output; an
authorization advisor can return a denial as ordinary text.

CLI models serialize conversation roles into task text; they do not provide
native system-message precedence, streaming, token usage, sampling controls,
media, provider output schemas, session resume, or `returnDirect` tools.
Unsupported options fail with `AiError`. Stdout is limited to 4 MiB.
Cancellation terminates the direct CLI process and closes the bridge; callback
cancellation is cooperative through `toolContext.signal`. Temporary MCP
configuration is removed, and the MCP child exits when its host disappears.
Other native descendants can outlive the direct CLI process.

Codex uses a read-only sandbox; Claude and Grok disable built-in tools.
AGY and Junie retain native permissions and may retain built-in tools. Grok
trusts the generated temporary workspace. Application prompts are not a sandbox.

In the playground's September 10, 2026 checks, AGY discovered MCP but headless
permissions denied execution; a server-scoped native permission rule is needed.
Junie reported HTTP 403, “No active JetBrains AI subscription found.”
These account restrictions are blocked runs, not successful validation.

Validation on September 10, 2026: Codex, Claude, and Grok completed sequential
live factory runs with the real plugin `Skill` callback; Grok also called
`Read`. The affected package/example suite passed 879 tests, and the final
focused suite passed 47 tests after additional limit and configuration coverage.
AI-package and playground TypeScript checks passed. A Node-targeted bundle
performed mocked HTTP inference without Bun. An extracted npm tarball outside
the checkout resolved and ran its packaged MCP child through a real stdio round
trip with a fake inference CLI. Hermes was not live-tested.
