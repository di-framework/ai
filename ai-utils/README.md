# @di-framework/ai-utils

`@huggingface/transformers@4.2.0` is an optional peer. Bun, npm, pnpm, and Yarn
installations do not fetch it for ordinary skills, small catalogs, or custom
`SkillEmbedder` implementations. Install it explicitly only when using the
default semantic embedder for a catalog above the indexing threshold.
`@di-framework/repo` is an optional peer used only when you pass a storage adapter
from that package into `SkillSearchConnection.fromStorageAdapter`.

Agentic extras for [`@di-framework/ai`](../di-framework-ai). **Agent Skills** (`SKILL.md`, progressive disclosure) plus jailed file tools, HITL questions, todos, optional web / memory / task, and opt-in `Write` / `Edit` / `Bash`.

## Repository instructions

`discoverAgentInstructions` loads hierarchical `AGENTS.md` files from the
workspace root through the working directory, returning combined
broad-to-specific content, ordered file provenance, and typed diagnostics:

```ts
import { discoverAgentInstructions } from '@di-framework/ai-utils';

const instructions = discoverAgentInstructions({
  workspace: process.cwd(),
  workingDirectory: 'packages/api',
  maxBytes: 32 * 1024,
});
```

Discovery never walks above the workspace and rejects symlink escapes. Pass
`allowedDirectories` to further restrict (not expand) the workspace boundary.
Whitespace-only files are skipped. A file that cannot fit wholly within the
combined UTF-8 byte limit is skipped with an
`instructions-max-bytes-exceeded` diagnostic; the default limit is 32 KiB.

Only `AGENTS.md` is automatic. Extra names are explicit fallbacks tried after
`AGENTS.md` at each directory, for example
`fallbackFilenames: ['TEAM_INSTRUCTIONS.md']`. `.agents.md` has no special
meaning. `.agents/AGENTS.md` participates naturally when the working directory
is inside `.agents/**`.

## Repository audit and migration

`auditAgentConfiguration` reports neutral instructions, skills, `.aiignore`
policy, suppressed sources, conflicts, and explicit vendor migration
opportunities without loading vendor configuration or changing files.

Turn that report into a deterministic, JSON-safe plan before opting into any
writes:

```ts
import {
  auditAgentConfiguration,
  executeAgentConfigurationMigration,
  planAgentConfigurationMigration,
} from '@di-framework/ai-utils';

const audit = auditAgentConfiguration({ workspace: process.cwd() });
const plan = planAgentConfigurationMigration(audit, {
  // Omit this list to plan every opportunity, or select exact audited sources.
  opportunityPaths: audit.migrationOpportunities.map(({ path }) => path),
  requests: [
    { target: '.agents/AGENTS.md', content: '# Agent defaults\n' },
    { target: '.agents/skills' },
    { target: '.aiignore', content: 'private/\n' },
  ],
});

// Dry-run is the default and never writes.
const preview = executeAgentConfigurationMigration(plan);

// Applying is an explicit opt-in and executes exactly the fingerprinted plan.
const result = executeAgentConfigurationMigration(plan, { dryRun: false });
```

Plans only target `AGENTS.md`, `.agents/AGENTS.md`, `.agents/skills/**`, and
`.aiignore`. Existing files become typed collisions unless replacement was
explicitly requested during planning. A replacement is a distinct
`replace-file` action and retains the old file beside the target with the
`.di-framework-backup` suffix. Creation uses same-directory staging and an
atomic no-replace link; source and target fingerprints are checked again at
execution time. Symbolic-link targets and sources are rejected.

Execution results classify every plan action as `applied`, `skipped`, or
`failed`, so partial failures are visible to automation. No vendor directory,
compatibility adapter, CLI, executable, or semantic index is created by these
APIs.

This is the TypeScript counterpart of [spring-ai-agent-utils](https://github.com/spring-ai-community/spring-ai-agent-utils). Skills run in your process. It is not Anthropic’s hosted Skills API.

Prefer **builders**: `SkillsAgent.builder()`, `SkillsToolbox.builder()`, `SkillsTool.builder()`, `SkillsIndex.builder()`. Free-function aliases remain. Skills stay in this package — `configureAi` / `@Agent` in `@di-framework/ai` are unchanged.

## Installation

```bash
bun add @di-framework/ai-utils @di-framework/ai @di-framework/core
```

Peer: `@di-framework/ai` and `@di-framework/core` (decorator DX uses core metadata; builders share the same install).

`@huggingface/transformers` is an optional peer used only by the default semantic indexer for large skill catalogs. Install it only when needed:

```bash
bun add @huggingface/transformers@4.2.0
```

## Quick start

```ts
import { OpenAiChatModel } from '@di-framework/ai';
import { SkillsAgent } from '@di-framework/ai-utils';

const agent = SkillsAgent.builder()
  .chatModel(new OpenAiChatModel({ model: 'gpt-4o-mini' }))
  .system('You help with TypeScript code review.')
  .instructionDiscovery({ workingDirectory: 'packages/api' })
  .addSkillsDirectory('.agents/skills')
  .workspace(process.cwd())
  .write()
  .shell()
  .build();

await agent.chat('Review src/UserController.ts');
```

Attach tools only:

```ts
import { ChatClient } from '@di-framework/ai';
import { SkillsToolbox } from '@di-framework/ai-utils';

const tools = SkillsToolbox.builder()
  .addSkillsDirectory('.agents/skills')
  .workspace(process.cwd())
  .buildTools();

const client = ChatClient.builder(model).defaultTools(...tools).build();
```

See [`examples/packages/ai-skills`](../../examples/packages/ai-skills) (`bun start` uses `process.env.OPENAI_API_KEY`) and [`examples/packages/ai-plugins`](../../examples/packages/ai-plugins) for official `@di-framework/plugin` discovery plus elective MCP wiring.

## What a skill is

A skill is a folder with a `SKILL.md` (YAML front matter + instructions). Optional `scripts/`, `references/`, and other files stay on disk until the model reads them.

```
.agents/skills/
└── code-reviewer/
    ├── SKILL.md
    ├── references/
    │   └── checklist.md
    └── scripts/
        └── count-lines.sh
```

```md
---
name: code-reviewer
description: Reviews TypeScript for nulls and framework conventions. Use when the user asks to review or audit code.
license: Apache-2.0
allowed-tools:
  - Read
  - Grep
metadata:
  author: you
---

# Code Reviewer

1. Read `references/checklist.md`
2. Check for null pointer risks
3. Suggest a concrete fix
```

Discovery puts only `name` + `description` in the `Skill` tool description. Activation returns the body plus the skill base directory. After activation, `allowed-tools` (if set) gates the rest of the toolbox, and file tools jail to workspace ∪ that skill folder.

Front matter is YAML (maps, lists, scalars, `|` / `>` blocks). `name` and `description` must satisfy [agentskills.io](https://agentskills.io/specification) rules; on disk the folder name must match `name`.

Validate definitions without constructing an agent, toolbox, or semantic index:

```ts
import {
  validateSkillCatalog,
  validateSkillDefinition,
  validateSkillDirectory,
  validateSkillsDirectory,
} from '@di-framework/ai-utils';

const directory = validateSkillDirectory('.agents/skills/code-reviewer');
const directoryCatalog = validateSkillsDirectory('.agents/skills');
const catalog = validateSkillCatalog({
  workspace: process.cwd(),
  directories: ['./team-skills'],
  sourceMode: 'merge',
});

if (!catalog.valid) {
  for (const diagnostic of catalog.diagnostics) {
    // Typed data only; choose presentation appropriate for your application.
    report(diagnostic.code, diagnostic.path, diagnostic.source, diagnostic.message);
  }
}
```

`validateSkillCatalog` calls the same source resolver as runtime discovery, so explicit, package, workspace, and user definitions have identical precedence. The first definition wins; same-source collisions are `skill-duplicate` warnings and lower-precedence definitions are `skill-shadowed` warnings. `validateResolvedSkillCatalog` accepts a previously resolved catalog, while `validateSkillDefinition` validates an in-memory `AgentSkill`. `validateSkillDirectory` checks one skill and its `SKILL.md`, referenced and bundled resources, readability, and symlinks; `validateSkillsDirectory` validates a complete directory catalog.

## Builders

| Factory | Builds | When to use |
| --- | --- | --- |
| `SkillsAgent.builder()` | `ChatAgent` + toolbox | Usual entry |
| `SkillsToolbox.builder()` | Toolbox (`build()` / `buildTools()`) | Attach to an existing `ChatClient` / `ChatAgent` |
| `SkillsTool.builder()` | `Skill` only | Tests or when you already have Read/Glob |
| `SkillsIndex.builder()` | Build-time compact semantic index | Large skill catalogs |

`.of(options)` on each factory matches the older options-object APIs. Aliases: `createSkillsAgent`, `createSkillsToolbox`, `skillsToolbox`, `skillsTool`.

## Decorator DX

Optional thin decorators map onto the builders above. They only store metadata; apply helpers call the existing builders. Prefer builders unless you want DI-style declarations.

| Decorator | Apply helper |
| --- | --- |
| `@Skills({ directories, packages, files, workspace, sourceMode })` | `skillsToolboxBuilderFrom` / `skillsAgentBuilderFrom` |
| `@SemanticSkillDiscovery({ indexFile, limit, ... })` | merged into `semanticDiscovery` (embedder/stores via overrides) |
| `@SkillsIndexConfig({ directories, threshold, retrievalLimit, ... })` | `skillsIndexBuilderFrom` then `.build()` |
| `@Skill({ name, description, content? })` | collected into catalog options |

```ts
import {
  SemanticSkillDiscovery,
  Skill,
  Skills,
  SkillsIndexConfig,
  skillsAgentFrom,
  skillsIndexBuilderFrom,
} from '@di-framework/ai-utils';

@Skills({ directories: ['.agents/skills'] })
@SemanticSkillDiscovery({ limit: 10 })
@Skill({ name: 'code-reviewer', description: 'Reviews TypeScript code.' })
class ApplicationSkills {}

const agent = skillsAgentFrom(ApplicationSkills, { chatModel: model });

@SkillsIndexConfig({ directories: ['.agents/skills'], threshold: 50 })
class ApplicationSkillsIndex {}

await skillsIndexBuilderFrom(ApplicationSkillsIndex).build();
```

Builders remain the supported escape hatch. Decorators do not run indexing or load Transformers.js.

### Shared toolbox methods

| Method | Effect |
| --- | --- |
| `addSkillsDirectory` / `addSkillsDirectories` | Load `SKILL.md` trees |
| `addSkillsFile` | Load one `SKILL.md` |
| `addSkill` / `addSkills` | In-memory skills |
| `addPackage` / `addPackages` | npm package or path (`package.json` `skills`, else `.agents/skills`, else `skills`) |
| `sourceMode('merge' \| 'replace')` | Supplement neutral defaults (default) or use only explicit sources |
| `workspace` | Default cwd / search root (also an allowed file root) |
| `extraAllowedDirectory(s)` | Extra sandbox roots |
| `write()` / `shell()` | Opt-in `Write`+`Edit` / `Bash` |
| `aiIgnore('discovery' \| 'read' \| 'read-write')` | Enforce root `.aiignore` for direct file tools |
| `confirmShell(fn)` | HITL gate for `Bash` |
| `askUser(handler)` | `AskUserQuestion` |
| `web(true \| { fetch, search, braveApiKey })` | `WebFetch` / `WebSearch` |
| `memories(true \| { directory })` | Long-term memory tools (`true` → `<workspace>/.memory`) |
| `task(true \| { chatModel, system })` | Nested `Task` subagent (`true` uses `.chatModel()`) |
| `glob` / `grep` / `list` / `todos` | Pass `false` to omit (on by default) |
| `perSkillSandbox(false)` | Keep all skill dirs allowed after activate |
| `toolName` / `toolDescriptionTemplate` / `onActivate` | Customize the `Skill` tool |

### `SkillsAgent.builder()` extras

`system`, `instructionDiscovery`, `chatModel`, `chatClient`, `extraTools`, `advisors`, `conversationMemory`, `defaultConversationId`, `defaultOptions`, `clientBuilderOptions`. `build()` returns `ChatAgent`; `buildBundle()` also returns the toolbox and the repository instruction discovery result.

Repository instruction discovery is enabled by default and is always bounded by
the agent's `workspace`. Configure hierarchy traversal with
`.instructionDiscovery({ workingDirectory, fallbackFilenames, maxBytes,
allowedDirectories })`, or disable it with `.instructionDiscovery(false)` (the
options-object equivalent is `instructionDiscovery: false`). Instruction files
only contribute text to the system prompt: they cannot add tools or expand file
tool roots.

System prompt sections are assembled deterministically in this order:

1. caller-provided `system` instructions;
2. repository instructions, ordered from the workspace root to the working
   directory; and
3. the memory-tool instructions when `memories` is enabled.

Caller instructions have the highest authority. Within repository instructions,
the file closest to the working directory is the most specific and takes
precedence over broader files. Memory instructions describe the configured
memory tools and do not override caller or repository policy. Use
`buildBundle().instructions` to inspect the exact ordered sources, combined
content, byte count, and diagnostics; it is `undefined` when discovery is
disabled.

## Toolbox tools

| Included by default | Opt-in |
| --- | --- |
| `Skill`, `Read`, `ListDirectory`, `Glob`, `Grep`, `TodoWrite` | `Write` / `Edit` (`.write()`), `Bash` (`.shell()`), `AskUserQuestion` (`.askUser()`), `WebFetch` / `WebSearch` (`.web()`), `MemoryView` / `MemoryWrite` / `MemoryEdit` / `MemoryDelete` / `MemoryRename` (`.memories()`), `Task` (`.task()`) |

File tools are limited to `workspace` ∪ skill folders (and extras). After a skill with `allowed-tools` activates, other tools are denied by name.

**`Bash` jails `cwd` only.** It is not a container: the process can still use the network or `cd` elsewhere. Prefer a container for untrusted skills. Use `confirmShell` when a human should approve commands.

## Discovery

### Shared source resolution

Operational features can resolve candidate files and directories before loading
their contents. `resolveAgentSources` is independent of agent construction and
CLI formatting:

```ts
import { resolveAgentSources } from '@di-framework/ai-utils';

const resolution = resolveAgentSources(
  [
    { path: '.agents/skills', origin: 'workspace', kind: 'directory' },
    { path: '~/.agents/skills', origin: 'user', kind: 'directory' },
  ],
  { workspace: process.cwd() },
);
```

Resolution is synchronous because it performs only local filesystem metadata
checks; it never reads source contents. Candidates are evaluated in array order
and every resolved source retains its `origin` and zero-based `precedence`. The
first occurrence of a real path wins;
later aliases and symlinks produce `source-duplicate` diagnostics. Missing,
unreadable, wrong-kind, broken-symlink, and boundary failures also have stable
typed diagnostic codes.

Relative paths resolve from `workspace`. Workspace sources cannot leave the
workspace, including through symlinks; user sources are similarly contained by
the home directory. Explicit, package, fallback, vendor, and migration sources
must remain within the workspace or an `allowedDirectories` root.

The only automatic roots are **`<workspace>/.agents/skills`** and
**`~/.agents/skills`**, in that order. `sourceMode` defaults to `merge`: explicit
directories and packages precede the workspace and user roots. In `replace`
mode, only explicit sources are used. The first definition of a skill name wins,
and `SkillsToolbox.skillDiagnostics` reports every shadowed definition with its
kept and ignored source paths. `SkillsToolbox.skillSources` retains each
resolved root's origin and precedence.

### Plugins

Plugins are filesystem bundles under `.agents/plugins/<id>/` (and
`~/.agents/plugins/<id>/`). Layout:

```
.agents/plugins/<plugin-id>/
  plugin.json          # required; name optional (defaults to directory name)
  mcp_config.json      # optional { mcpServers: ... }
  hooks.json           # optional object
  skills/              # optional nested SKILL.md trees
  rules/               # optional *.md rule files
```

`SkillsAgent` / `SkillsToolbox` do **not** load plugins by themselves. Wiring is
one of three modes:

#### Automatic (discovery only)

Calling the plugin APIs with no explicit roots uses the same merge/replace model
as skills. Defaults are **`<workspace>/.agents/plugins`** then
**`~/.agents/plugins`**. Explicit `directories` / `packages` and `sourceMode`
work like `resolveSkillSources`. A package that publishes `plugin.json` at its
own root (for example `@di-framework/plugin`) resolves as a single-plugin source.
Nested `skills/` are validated with the skill catalog rules when you validate a
plugin catalog.

```ts
import { resolvePluginSources, validatePluginCatalog } from '@di-framework/ai-utils';

const resolution = resolvePluginSources({ workspace: process.cwd() });
const catalog = validatePluginCatalog({ workspace: process.cwd() });
const fromPackage = validatePluginCatalog({
  packages: ['@di-framework/plugin'],
  sourceMode: 'replace',
});
```

Automatic here means **where plugins are found**, not that agents consume them.

#### Programmatic

Load or validate a specific bundle (or an already-resolved catalog) without
depending on defaults:

```ts
import {
  loadPluginDirectory,
  loadPluginsDirectory,
  validatePluginDirectory,
  validateResolvedPluginCatalog,
} from '@di-framework/ai-utils';

const plugin = loadPluginDirectory('.agents/plugins/di-framework');
const fromRoot = loadPluginsDirectory('.agents/plugins');
const one = validatePluginDirectory('.agents/plugins/di-framework');
```

`AgentPlugin` exposes `skillsDirectory`, `mcpConfig`, `rules`, and `hooks` for
callers to consume. This package does not start MCP servers or run hooks.

#### Elective (opt into an agent)

Feed plugin pieces into the skills / instructions stack yourself. Typical
pattern: validate the catalog, then pass each plugin’s `skills/` root into
skill discovery and append rule markdown to system context:

```ts
import {
  SkillsAgent,
  validatePluginCatalog,
} from '@di-framework/ai-utils';

const workspace = process.cwd();
const catalog = validatePluginCatalog({ workspace });
if (!catalog.valid) {
  throw new Error(catalog.diagnostics.map((d) => d.message).join('\n'));
}

const pluginSkillDirs = catalog.plugins
  .map((plugin) => plugin.skillsDirectory)
  .filter((path): path is string => path != null);

const ruleText = catalog.plugins
  .flatMap((plugin) => plugin.rules)
  .map((rule) => rule.content)
  .join('\n\n');

const agent = SkillsAgent.builder()
  .chatModel(model)
  .workspace(workspace)
  .addSkillsDirectories(pluginSkillDirs)
  .system(
    ruleText
      ? `You help with this codebase.\n\n${ruleText}`
      : 'You help with this codebase.',
  )
  .build();
```

MCP is elective the same way: read `plugin.mcpConfig.mcpServers` and register
those entries with whatever MCP client your process uses. Hooks stay data until
you interpret `hooks.json`.

### `.aiignore` policy

Root-workspace `.aiignore` rules can be loaded and evaluated independently of
agent construction, tools, and CLI behavior:

```ts
import { evaluateAiIgnorePath, loadAiIgnorePolicy } from '@di-framework/ai-utils';

const policy = loadAiIgnorePolicy({ workspace: process.cwd() });
const evaluation = evaluateAiIgnorePath(policy, 'build/generated.ts');

if (!evaluation.pathAccess.ok) {
  // The filesystem sandbox denied the path before policy evaluation.
} else if (evaluation.ignored) {
  console.log(evaluation.rule?.source.path, evaluation.rule?.line);
}
```

The syntax follows `.gitignore` conventions: comments, `*`, `**`, `?`,
character ranges, directory-only and root-relative patterns, `!` negation, and
last-match-wins precedence. Evaluations report the effective rule and policy
source. `compileAiIgnorePolicy` provides the same behavior for explicitly
supplied policy text.

Only `<workspace>/.aiignore` is discovered; nested policy files are not loaded.
The root policy file is always returned as `policy-file`, even if a rule would
otherwise ignore it. `.aiignore` does not replace `.gitignore`, and its decisions
cannot grant access denied by the filesystem sandbox. Lexical escapes and
symlinks that leave the workspace are denied before rules are considered.

Discovery uses the root policy consistently. `SkillsToolbox` loads it from the
configured workspace and applies it to recursive Glob/Grep walking and skill
catalog discovery; `discoverAgentInstructions` applies it before reading each
instruction source. Ignored directories are pruned before their entries are
visited. Direct low-level factories accept an `aiIgnorePolicy`, and
`onSuppressed` receives content-free `aiignore-suppressed` diagnostics:

```ts
const suppressed = [];
const policy = loadAiIgnorePolicy({ workspace: process.cwd() });
const tools = SkillsToolbox.builder()
  .workspace(process.cwd())
  .aiIgnorePolicy(policy)
  .onSuppressed((diagnostic) => suppressed.push(diagnostic))
  .buildTools();

const instructions = discoverAgentInstructions({
  workspace: process.cwd(),
  aiIgnorePolicy: policy,
});
```

Diagnostics contain the suppressed path, path kind, discovery surface, policy
path, and matching line number, but never the suppressed file content or rule
text. The fixed `node_modules`, `.git`, `dist`, and `coverage` traversal skips,
maximum depth, allowed-directory checks, and filesystem sandbox remain stronger
than `.aiignore`; a negated rule cannot re-enable them. This section covers
discovery only. Enforcement for direct Read/Write/Edit/List access is configured
separately.

```ts
const isolated = SkillsToolbox.builder()
  .addSkillsDirectory('./team-skills')
  .sourceMode('replace')
  .workspace(process.cwd())
  .build();
```

Direct file tools accept a compiled policy and an explicit enforcement mode:

```ts
import { loadAiIgnorePolicy, readTool } from '@di-framework/ai-utils';

const policy = loadAiIgnorePolicy({ workspace: process.cwd() });
const read = readTool({
  allowedDirectories: [process.cwd()],
  aiIgnore: { policy, enforcement: 'read' },
});
```

Toolboxes and agents can load the root policy automatically with
`.aiIgnore(mode)`. Enforcement is opt-in and cumulative:

| Mode | Directory listing | Read | Edit | Write |
| --- | --- | --- | --- | --- |
| `discovery` | Filter/reject | Allow | Allow | Allow |
| `read` | Filter/reject | Reject | Reject | Allow |
| `read-write` | Filter/reject | Reject | Reject | Reject |

Only paths matched as ignored are rejected. Negated rules can include a path,
but cannot override the filesystem sandbox or allowed-directory boundaries,
which are checked first. Policy rejections report the target path, policy path,
and matching rule line without including policy or file contents.

`addPackage('@scope/pack')` resolves `pack/package.json` from the workspace,
then uses its `skills` field (string or string[]). Without that field, it tries
`.agents/skills`, then `skills` under the package root.

### Large catalogs

By default, every skill name and description is placed in the `Skill` tool. For catalogs above the default threshold of 50, build a local semantic index instead:

```bash
di-framework skills index build --skills-dir .agents/skills
```

The equivalent programmatic build API is:

```ts
import { SkillsIndex } from '@di-framework/ai-utils';

await SkillsIndex.builder()
  .addSkillsDirectory('.agents/skills')
  .build();
```

This writes `.di-framework/skills-index.json` and a content-verified
`.vectors.bin` sidecar. Put the command in `prebuild` (or call the API from a
build script), then enable fail-closed runtime retrieval:

```ts
const agent = SkillsAgent.builder()
  .chatModel(model)
  .addSkillsDirectory('.agents/skills')
  .semanticDiscovery()
  .build();
```

The default index is also detected automatically when present. Explicit `.semanticDiscovery()` throws if it is missing. At or below 50 skills, the build writes metadata only and keeps normal full-catalog discovery.

The default indexer uses the optional `@huggingface/transformers` peer locally. It is loaded only after the catalog exceeds the threshold; small catalogs and custom `SkillEmbedder` implementations do not require it. The indexer tokenizes each exact `SKILL.md` into 256-token chunks with 32-token overlap, embeds them with a pinned quantized BGE model, and writes format-v3 vectors with per-vector symmetric int8 quantization. The manifest records dimensions, scales, norms, byte length, and a SHA-256 sidecar hash. Version-2 float32/base64 JSONL remains readable. Runtime scores directly over int8 storage and sends only the top 10 names/descriptions to the chat model. Chunk text and vectors never enter the prompt; the chosen body still loads only after `Skill` activation.

Format v3 also stores compact BM25 postings built from each skill name,
description, and body. Retrieval deterministically fuses lexical and dense ranks,
pins explicit skill names, and returns at most one match per skill. The manifest
records all BM25, reciprocal-rank-fusion, and abstention parameters. A valid
abstention removes the `Skill` tool for that request; missing, stale, or corrupt
indexes remain hard errors and never fall back to the full catalog.

### Index operations

The package exports CLI-independent functions for every index workflow. They accept explicit options,
return typed results, and never parse arguments, print output, or terminate the process:

```ts
import {
  buildSkillsIndex,
  inspectSkillsIndex,
  migrateSkillsIndex,
  querySkillsIndex,
  SkillsIndexOperationError,
  validateSkillsIndex,
} from '@di-framework/ai-utils';

const built = await buildSkillsIndex({
  directories: ['.agents/skills'],
  outputFile: '.di-framework/skills-index.json',
  onProgress(completed, total) {
    reportEmbeddingProgress(completed, total);
  },
});

const inspection = inspectSkillsIndex({ inputFile: built.outputFile });
const validation = validateSkillsIndex({
  inputFile: built.outputFile,
  directories: ['.agents/skills'],
});
const query = await querySkillsIndex({
  inputFile: built.outputFile,
  query: 'review row-level security',
});
const migration = migrateSkillsIndex({
  inputFile: 'old-index.jsonl',
  outputFile: built.outputFile,
});
```

`inspectSkillsIndex` reports format/model metadata, integrity sizes, scoring configuration, load
latency, and memory. `validateSkillsIndex` reports source drift as a typed negative result (`valid:
false`) while corrupt or unreadable indexes throw. `querySkillsIndex` returns safe names and
descriptions, dense and lexical score components, matched chunk numbers, and load/embed/search timing.
`migrateSkillsIndex` reads version 2 or 3 and explicitly writes version 3. Bodies and vector contents
are not included in any result.

The non-build operations accept `onProgress(event)` with typed operation and phase values. Build keeps
its existing chunk-oriented `onProgress(completed, total)` callback. Operational failures throw
`SkillsIndexOperationError`, whose stable `operation` and `code` fields let callers distinguish invalid
options, missing sources or indexes, invalid indexes, embedding failures, write failures, and unexpected
operation failures. CLI layers are responsible for presentation and exit-status mapping.

Configure `.threshold()`, `.chunkTokens()`, `.chunkOverlapTokens()`, `.retrievalLimit()`, or `.embedder()` on `SkillsIndex.builder()` when the defaults do not fit the corpus. The `buildSkillsIndex(options)` free-function alias remains available. Runtime `.semanticDiscovery({ limit, minScore, embedder })` can override candidate count and query embedding.

Platform runtimes can independently supply `catalogStore`, `vectorSearch`, and a build-time
`writer`. Wire a durable backend with `SkillSearchRepository(connection)` and
`SkillSearchIndexer(data, connection)`, where `SkillSearchConnection.fromVectorStore` accepts any
`VectorStore` (including `BunSqliteVectorStore`) and `fromStorageAdapter` accepts
`@di-framework/repo` adapters. Remote catalogs use `buildAsync()` / `createSkillsAgentAsync()` so
discovery lists only descriptors and the selected body is fetched after activation. See the
[skill adapter authoring guide](../../docs/skill-adapter-authoring.md) for contracts, readiness,
failure behavior, contract tests, and performance reporting.

## Skill-only and MCP

```ts
import { agentSkill, SkillsTool, skillsToolboxAsMcp } from '@di-framework/ai-utils';

const skillOnly = SkillsTool.builder()
  .addSkill(
    agentSkill({
      name: 'xlsx',
      description: 'Build spreadsheets when asked for a spreadsheet.',
      content: 'Prefer a real .xlsx over a description.',
    }),
  )
  .build();

const mcp = skillsToolboxAsMcp({
  directories: ['.agents/skills'],
  workspace: process.cwd(),
});
```

`skillsToolboxAsMcp` returns `{ descriptor, handler, returnDirect }[]` from `@di-framework/ai` `toolCallbackAsMcpTool`.

## Safety

- Path jail rejects raw `..`, sibling prefix matches, and escaping symlinks.
- `.aiignore` prunes Glob, Grep, skill, and `AGENTS.md` discovery after sandbox checks.
- `Write` / `Edit` / `Bash` are off until you enable them.
- `allowed-tools` is enforced only after `Skill` runs.
- `WebSearch` needs `BRAVE_API_KEY` or `.web({ braveApiKey })`.
- `Task` subagents do not get `AskUserQuestion` or nested `Task`.

## API

| Export | Role |
| --- | --- |
| `SkillsAgent` / `SkillsAgentBuilder` | Preferred agent factory |
| `SkillsToolbox` / `SkillsToolboxBuilder` | Preferred toolbox factory |
| `SkillsTool` / `SkillsToolBuilder` | `Skill` only |
| `SkillsIndex` / `SkillsIndexBuilder` | Build-time semantic skill index |
| `skillsToolboxAsMcp` | MCP descriptors + handlers |
| File / shell | `readTool`, `listDirectoryTool`, `globTool`, `grepTool`, `writeTool`, `editTool`, `bashTool` |
| Agent extras | `todoWriteTool`, `askUserQuestionTool`, `webFetchTool`, `webSearchTool`, `memoryTools`, `taskTool` |
| Discovery | `discoverAgentInstructions`, `loadAiIgnorePolicy`, `evaluateAiIgnorePath`, `loadSkillsDirectory`, `resolveSkillPackageDirectories`, `existingSkillDirectories`, `resolvePluginSources`, `loadPluginDirectory`, `loadPluginsDirectory`, `resolvePluginPackageDirectories`, `existingPluginDirectories`, `SkillsIndex.builder()`, `searchSkillsIndex`, `di-framework skills index` commands |
| Parse / validate | `parseSkillMarkdown`, `parseYaml`, `agentSkill`, `validateSkill`, `validateSkillDefinition`, `validateSkillDirectory`, `validateSkillsDirectory`, `validateSkillCatalog`, `validateResolvedSkillCatalog`, `parsePluginManifest`, `parseMcpConfig`, `validatePlugin`, `validatePluginDefinition`, `validatePluginDirectory`, `validatePluginsDirectory`, `validatePluginCatalog`, `validateResolvedPluginCatalog` |

**Style:** `static of` / `static builder` and free functions for pure helpers. See [docs/static-methods-convention.md](../../docs/static-methods-convention.md).
