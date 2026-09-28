import {
  type Advisor,
  ChatAgent,
  type ChatClient,
  type ChatClientBuilderOptions,
  type ChatMemory,
  type ChatModel,
  type ChatOptions,
  type ToolSource,
} from '@di-framework/ai';
import {
  type DiscoverAgentInstructionsOptions,
  type DiscoverAgentInstructionsResult,
  discoverAgentInstructions,
} from '../instructions/discover-agent-instructions.ts';
import { formatMemorySystemPrompt } from '../tools/memory-tools.ts';
import { SkillsFluent } from './skills-fluent.ts';
import {
  createSkillsToolbox,
  createSkillsToolboxAsync,
  type SkillsToolbox,
  type SkillsToolboxOptions,
} from './skills-toolbox.ts';

export type SkillsAgentInstructionDiscoveryOptions = Omit<
  DiscoverAgentInstructionsOptions,
  'workspace'
>;

export interface CreateSkillsAgentOptions extends SkillsToolboxOptions {
  readonly chatModel?: ChatModel;
  readonly chatClient?: ChatClient;
  readonly system?: string;
  readonly extraTools?: readonly ToolSource[];
  readonly defaultOptions?: ChatOptions;
  readonly advisors?: readonly Advisor[];
  readonly conversationMemory?: ChatMemory;
  readonly defaultConversationId?: string;
  readonly builder?: ChatClientBuilderOptions;
  /** Repository instruction discovery is enabled by default; pass false to disable it. */
  readonly instructionDiscovery?: false | SkillsAgentInstructionDiscoveryOptions;
}

export interface SkillsAgentBundle {
  readonly agent: ChatAgent;
  readonly toolbox: SkillsToolbox;
  /** Loaded repository instruction content, provenance, and diagnostics; absent when disabled. */
  readonly instructions?: DiscoverAgentInstructionsResult;
}

export class SkillsAgentBuilder extends SkillsFluent<SkillsAgentBuilder> {
  private systemText?: string;
  private extraToolSources?: readonly ToolSource[];
  private client?: ChatClient;
  private chatOptions?: ChatOptions;
  private advisorList?: readonly Advisor[];
  private memory?: ChatMemory;
  private conversationId?: string;
  private clientBuilder?: ChatClientBuilderOptions;
  private instructionDiscoveryOptions?: false | SkillsAgentInstructionDiscoveryOptions;

  system(text: string): this {
    this.systemText = text;
    return this;
  }

  extraTools(...tools: ToolSource[]): this {
    this.extraToolSources = [...(this.extraToolSources ?? []), ...tools];
    return this;
  }

  chatClient(client: ChatClient): this {
    this.client = client;
    return this;
  }

  defaultOptions(options: ChatOptions): this {
    this.chatOptions = options;
    return this;
  }

  advisors(...advisors: Advisor[]): this {
    this.advisorList = [...(this.advisorList ?? []), ...advisors];
    return this;
  }

  conversationMemory(memory: ChatMemory): this {
    this.memory = memory;
    return this;
  }

  defaultConversationId(id: string): this {
    this.conversationId = id;
    return this;
  }

  clientBuilderOptions(options: ChatClientBuilderOptions): this {
    this.clientBuilder = options;
    return this;
  }

  instructionDiscovery(options: false | SkillsAgentInstructionDiscoveryOptions = {}): this {
    this.instructionDiscoveryOptions = options;
    return this;
  }

  build(): ChatAgent {
    return this.buildBundle().agent;
  }

  async buildAsync(): Promise<ChatAgent> {
    return (await this.buildBundleAsync()).agent;
  }

  buildBundle(): SkillsAgentBundle {
    return createSkillsAgentBundle(this.toAgentOptions());
  }

  buildBundleAsync(): Promise<SkillsAgentBundle> {
    return createSkillsAgentBundleAsync(this.toAgentOptions());
  }

  toAgentOptions(): CreateSkillsAgentOptions {
    return {
      ...this.toOptions(),
      system: this.systemText,
      extraTools: this.extraToolSources,
      chatClient: this.client,
      defaultOptions: this.chatOptions,
      advisors: this.advisorList,
      conversationMemory: this.memory,
      defaultConversationId: this.conversationId,
      builder: this.clientBuilder,
      instructionDiscovery: this.instructionDiscoveryOptions,
    };
  }
}

/**
 * Preferred factory: {@code SkillsAgent.builder().chatModel(model).addSkillsDirectory(...).build()}.
 */
export const SkillsAgent = {
  builder(): SkillsAgentBuilder {
    return new SkillsAgentBuilder();
  },
  of(options: CreateSkillsAgentOptions): ChatAgent {
    return createSkillsAgent(options);
  },
  ofAsync(options: CreateSkillsAgentOptions): Promise<ChatAgent> {
    return createSkillsAgentAsync(options);
  },
};

/**
 * ChatAgent with {@link createSkillsToolbox} attached. Skills stay in this
 * package — {@code configureAi} / {@code @Agent} are unchanged.
 * Prefer {@link SkillsAgent.builder}.
 */
export function createSkillsAgent(options: CreateSkillsAgentOptions): ChatAgent {
  return createSkillsAgentBundle(options).agent;
}

export async function createSkillsAgentAsync(
  options: CreateSkillsAgentOptions,
): Promise<ChatAgent> {
  return (await createSkillsAgentBundleAsync(options)).agent;
}

export function createSkillsAgentBundle(options: CreateSkillsAgentOptions): SkillsAgentBundle {
  const toolbox = createSkillsToolbox(options);
  return assembleSkillsAgent(options, toolbox);
}

export async function createSkillsAgentBundleAsync(
  options: CreateSkillsAgentOptions,
): Promise<SkillsAgentBundle> {
  return assembleSkillsAgent(options, await createSkillsToolboxAsync(options));
}

function assembleSkillsAgent(
  options: CreateSkillsAgentOptions,
  toolbox: SkillsToolbox,
): SkillsAgentBundle {
  const workspace = options.workspace ?? process.cwd();
  const instructions =
    options.instructionDiscovery === false
      ? undefined
      : discoverAgentInstructions({
          ...options.instructionDiscovery,
          workspace,
          workingDirectory: options.instructionDiscovery?.workingDirectory ?? workspace,
        });
  const memoriesDir =
    options.memories === true
      ? `${workspace.replace(/[/\\]$/, '')}/.memory`
      : options.memories && typeof options.memories === 'object'
        ? options.memories.directory
        : undefined;
  const systemParts = [
    options.system,
    instructions?.content,
    memoriesDir ? formatMemorySystemPrompt(memoriesDir) : undefined,
  ].filter((part): part is string => Boolean(part?.trim()));

  const agent = ChatAgent.create({
    chatModel: options.chatModel,
    chatClient: options.chatClient,
    system: systemParts.length > 0 ? systemParts.join('\n\n') : undefined,
    tools: [...toolbox.tools, ...(options.extraTools ?? [])],
    defaultOptions: options.defaultOptions,
    advisors: [toolbox.retrievalAdvisor, ...(options.advisors ?? [])].filter(
      (advisor): advisor is Advisor => advisor != null,
    ),
    memory: options.conversationMemory,
    defaultConversationId: options.defaultConversationId,
    builder: options.builder,
  });
  return { agent, toolbox, instructions };
}
