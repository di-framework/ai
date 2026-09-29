import type { ChatModel } from '@di-framework/ai';
import type { AiIgnorePolicy, AiIgnoreSuppressionDiagnostic } from '../policy/index.ts';
import type { AiIgnoreEnforcement } from '../tools/aiignore-enforcement.ts';
import type { QuestionHandler } from '../tools/ask-user-question-tool.ts';
import type { BashConfirmInput } from '../tools/bash-tool.ts';
import type { AgentSkill } from './parse-skill-markdown.ts';
import type { SkillSourceMode } from './resolve-skill-sources.ts';
import type {
  SkillsSemanticDiscoveryOptions,
  SkillsToolboxMemoriesOptions,
  SkillsToolboxOptions,
  SkillsToolboxTaskOptions,
  SkillsToolboxWebOptions,
} from './skills-toolbox.ts';

/**
 * Shared fluent setters for {@link SkillsToolboxBuilder} and {@link SkillsAgentBuilder}.
 */
export class SkillsFluent<T extends SkillsFluent<T>> {
  protected readonly draft: {
    -readonly [K in keyof SkillsToolboxOptions]?: SkillsToolboxOptions[K];
  } = {};

  protected self(): T {
    return this as unknown as T;
  }

  addSkill(skill: AgentSkill): T {
    this.draft.skills = [...(this.draft.skills ?? []), skill];
    return this.self();
  }

  addSkills(skills: readonly AgentSkill[]): T {
    this.draft.skills = [...(this.draft.skills ?? []), ...skills];
    return this.self();
  }

  addSkillsDirectory(directory: string): T {
    this.draft.directories = [...(this.draft.directories ?? []), directory];
    return this.self();
  }

  addSkillsDirectories(directories: readonly string[]): T {
    this.draft.directories = [...(this.draft.directories ?? []), ...directories];
    return this.self();
  }

  addSkillsFile(skillMdPath: string): T {
    this.draft.files = [...(this.draft.files ?? []), skillMdPath];
    return this.self();
  }

  addPackage(spec: string): T {
    this.draft.packages = [...(this.draft.packages ?? []), spec];
    return this.self();
  }

  addPackages(specs: readonly string[]): T {
    this.draft.packages = [...(this.draft.packages ?? []), ...specs];
    return this.self();
  }

  /** Choose whether explicit sources supplement or replace neutral defaults. */
  sourceMode(mode: SkillSourceMode): T {
    this.draft.sourceMode = mode;
    return this.self();
  }

  workspace(path: string): T {
    this.draft.workspace = path;
    return this.self();
  }

  userDirectory(path: string): T {
    this.draft.userDirectory = path;
    return this.self();
  }

  aiIgnorePolicy(policy: AiIgnorePolicy): T {
    this.draft.aiIgnorePolicy = policy;
    return this.self();
  }

  onSuppressed(handler: (diagnostic: AiIgnoreSuppressionDiagnostic) => void): T {
    this.draft.onSuppressed = handler;
    return this.self();
  }

  extraAllowedDirectory(path: string): T {
    this.draft.extraAllowedDirectories = [...(this.draft.extraAllowedDirectories ?? []), path];
    return this.self();
  }

  extraAllowedDirectories(paths: readonly string[]): T {
    this.draft.extraAllowedDirectories = [...(this.draft.extraAllowedDirectories ?? []), ...paths];
    return this.self();
  }

  toolName(name: string): T {
    this.draft.toolName = name;
    return this.self();
  }

  toolDescriptionTemplate(template: string): T {
    this.draft.toolDescriptionTemplate = template;
    return this.self();
  }

  onActivate(handler: (skill: AgentSkill) => void): T {
    this.draft.onActivate = handler;
    return this.self();
  }

  glob(enabled = true): T {
    this.draft.glob = enabled;
    return this.self();
  }

  grep(enabled = true): T {
    this.draft.grep = enabled;
    return this.self();
  }

  list(enabled = true): T {
    this.draft.list = enabled;
    return this.self();
  }

  write(enabled = true): T {
    this.draft.write = enabled;
    return this.self();
  }

  aiIgnore(enforcement: AiIgnoreEnforcement): T {
    this.draft.aiIgnore = enforcement;
    return this.self();
  }

  shell(enabled = true): T {
    this.draft.shell = enabled;
    return this.self();
  }

  shellTimeoutMs(timeoutMs: number): T {
    this.draft.shellTimeoutMs = timeoutMs;
    return this.self();
  }

  confirmShell(confirm: (input: BashConfirmInput) => boolean | Promise<boolean>): T {
    this.draft.confirmShell = confirm;
    return this.self();
  }

  todos(enabled = true): T {
    this.draft.todos = enabled;
    return this.self();
  }

  askUser(handler: QuestionHandler): T {
    this.draft.askUser = handler;
    return this.self();
  }

  web(enabledOrOptions: boolean | SkillsToolboxWebOptions): T {
    this.draft.web = enabledOrOptions;
    return this.self();
  }

  memories(enabledOrOptions: boolean | SkillsToolboxMemoriesOptions): T {
    this.draft.memories = enabledOrOptions;
    return this.self();
  }

  task(enabledOrOptions: boolean | SkillsToolboxTaskOptions): T {
    this.draft.task = enabledOrOptions;
    return this.self();
  }

  chatModel(model: ChatModel): T {
    this.draft.chatModel = model;
    return this.self();
  }

  perSkillSandbox(enabled = true): T {
    this.draft.perSkillSandbox = enabled;
    return this.self();
  }

  semanticDiscovery(enabledOrOptions: boolean | SkillsSemanticDiscoveryOptions = true): T {
    this.draft.semanticDiscovery = enabledOrOptions;
    return this.self();
  }

  toOptions(): SkillsToolboxOptions {
    return { ...this.draft };
  }
}
