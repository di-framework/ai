# di-framework/ai

TypeScript libraries for building AI applications with chat models, tools, memory, retrieval, MCP, and agents. Use the fluent APIs directly or integrate them with `@di-framework/core` dependency injection.

| Package | Purpose | Documentation |
| --- | --- | --- |
| `@di-framework/ai` | Chat clients, providers, tools, memory, RAG, MCP, workflows, and A2A | [AI README](ai/README.md) |
| `@di-framework/ai-utils` | Agent Skills (`SKILL.md`), repository instructions, file and shell tools, and skill discovery | [AI utilities README](ai-utils/README.md) |

## Installation

The examples use Bun to run TypeScript directly. Both packages export TypeScript source; other runtimes need compatible TypeScript tooling.

```sh
bun add @di-framework/ai@^6 @di-framework/core@^5 @di-framework/auth@^5
bun add --dev typescript@^5

# Add skills and agent utilities when needed:
bun add @di-framework/ai-utils@^6
```

You can also install these packages with npm. The first release from this repository is **6.0.0**; the core and auth peers remain on published **5.x** packages.

## Quick start

Save this as `chat.ts`:

```ts
import { ChatClient, createChatModel } from '@di-framework/ai';

const client = ChatClient.builder(createChatModel())
  .defaultSystem('You are a concise, helpful assistant.')
  .build();

const answer = await client
  .prompt('Explain dependency injection in two sentences.')
  .call()
  .content();

console.log(answer);
```

Set your provider's API key in the environment, then run:

```sh
# Requires OPENAI_API_KEY:
PROVIDER=openai AUTH=api bun chat.ts

# Or, with ANTHROPIC_API_KEY:
PROVIDER=anthropic AUTH=api bun chat.ts
```

`createChatModel()` reads `PROVIDER`, `AUTH`, and optional `MODEL`. You can select them explicitly with `createChatModel({ provider: 'anthropic', auth: 'api', model: 'your-model-id' })`. See the [provider documentation](ai/README.md#select-api-or-subscription-access) for supported routes and subscription setup.

### Stream a response

Using the same `client`, replace the call above with:

```ts
for await (const content of client.prompt('Explain dependency injection.').stream().content()) {
  console.log(content);
}
```

Each value contains the accumulated response text so far, suitable for replacing the displayed answer as it arrives.

## Give the model a tool

Describe a callback with a JSON Schema. The client handles tool requests and feeds results back to the model before returning the final answer.

```ts
import { ChatClient, createChatModel, functionToolCallback } from '@di-framework/ai';

const add = functionToolCallback<{ a: number; b: number }, number>({
  name: 'add',
  description: 'Add two numbers.',
  inputSchema: {
    type: 'object',
    properties: { a: { type: 'number' }, b: { type: 'number' } },
    required: ['a', 'b'],
    additionalProperties: false,
  },
  call: ({ a, b }) => a + b,
});

const client = ChatClient.builder(createChatModel()).defaultTools(add).build();
console.log(await client.prompt('Use add to calculate 19 + 23.').call().content());
```

Run this and the following model-backed examples with the same provider environment as the quick start.

## Keep conversation memory

`ChatAgent` wraps a client with reusable instructions, tools, and optional memory. Supply the same conversation ID for related turns and a different ID for each independent conversation.

```ts
import { ChatAgent, createChatModel, MessageWindowChatMemory } from '@di-framework/ai';

const agent = ChatAgent.create({
  chatModel: createChatModel(),
  system: 'You are a helpful assistant.',
  memory: MessageWindowChatMemory.builder().maxMessages(20).build(),
});

await agent.chat('My name is Ada.', { conversationId: 'session-1' });
const reply = await agent.chat('What is my name?', { conversationId: 'session-1' });
console.log(reply.content);
```

This memory lives in the current process. The message window drops older turns as it fills.

## Add Agent Skills

Install `@di-framework/ai-utils`, then create `.agents/skills/code-reviewer/SKILL.md` in your project:

```md
---
name: code-reviewer
description: Review TypeScript code for correctness. Use when asked to review code.
---

# Code review

1. Read the requested file.
2. Identify correctness issues and explain their impact.
3. Suggest a concrete fix for each issue.
```

Run this from the project directory, replacing `src/index.ts` with a file to review:

```ts
import { createChatModel } from '@di-framework/ai';
import { SkillsAgent } from '@di-framework/ai-utils';

const agent = SkillsAgent.builder()
  .chatModel(createChatModel())
  .system('Help review TypeScript code. Use the code-reviewer skill when reviewing.')
  .workspace(process.cwd())
  .addSkillsDirectory('.agents/skills')
  .build();

const reply = await agent.chat('Review src/index.ts.');
console.log(reply.content);
```

Skills expose descriptions first and load their full instructions when activated. The toolbox includes file-reading tools; writing, editing, and shell execution are opt-in. See the [utilities documentation](ai-utils/README.md) for toolbox configuration, instruction discovery, and skill catalogs.

## Try the client without an API key

Use `FakeChatModel` for a fixed response, or `ScriptedChatModel` for multi-turn and tool-calling tests:

```ts
import { ChatClient, FakeChatModel } from '@di-framework/ai';

const model = new FakeChatModel('Hello, Ada!');
const client = ChatClient.create(model);

console.log(await client.prompt('Say hello to Ada.').call().content());
// Hello, Ada!
console.log(model.calls.length);
// 1
```

## Development

From the repository root:

```sh
bun install
bun test
bun run typecheck
bun run lint
```

Run a package's tests with `bun test ai/tests` or `bun test ai-utils/tests`. The workspace uses published core/auth peers. The CLI host (`di-framework agent`, `di-framework skills`) lives in the separate `di-framework/di-framework` repository.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
