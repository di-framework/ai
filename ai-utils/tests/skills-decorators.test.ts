import { describe, expect, test } from 'bun:test';
import {
  type Advisor,
  ChatClient,
  MessageWindowChatMemory,
  ScriptedChatModel,
} from '@di-framework/ai';
import {
  agentSkill,
  getDeclaredSkills,
  getSemanticSkillDiscoveryMetadata,
  getSkillsIndexMetadata,
  getSkillsMetadata,
  SemanticSkillDiscovery,
  Skill,
  type SkillEmbedder,
  Skills,
  SkillsAgent,
  SkillsIndex,
  SkillsIndexConfig,
  SkillsToolbox,
  type SkillTokenChunkOptions,
  skillsAgentBuilderFrom,
  skillsAgentFrom,
  skillsIndexBuilderFrom,
  skillsToolboxBuilderFrom,
  skillsToolboxFrom,
  skillsToolboxOptionsFrom,
} from '../src/index.ts';

class FakeEmbedder implements SkillEmbedder {
  readonly id = 'fake@test';
  readonly model = 'fake';
  readonly revision = 'test';

  async embed(texts: readonly string[]): Promise<readonly Float32Array[]> {
    return texts.map(() => new Float32Array([1, 0, 0]));
  }

  async split(text: string, _options: SkillTokenChunkOptions): Promise<readonly string[]> {
    return [text];
  }
}

const passthroughAdvisor: Advisor = {
  name: 'passthrough',
  order: 0,
};

describe('skills decorators', () => {
  test('Skills and SemanticSkillDiscovery store metadata', () => {
    @Skills({
      directories: ['.agents/skills'],
      packages: ['@company/skills'],
      workspace: '/tmp/app',
      sourceMode: 'replace',
    })
    @SemanticSkillDiscovery({
      indexFile: '.di-framework/skills-index.jsonl',
      limit: 10,
      minScore: 0.2,
    })
    class ApplicationSkills {}

    expect(getSkillsMetadata(ApplicationSkills)).toEqual({
      directories: ['.agents/skills'],
      packages: ['@company/skills'],
      workspace: '/tmp/app',
      sourceMode: 'replace',
    });
    expect(getSemanticSkillDiscoveryMetadata(ApplicationSkills)).toEqual({
      indexFile: '.di-framework/skills-index.jsonl',
      limit: 10,
      minScore: 0.2,
    });
    expect(getSkillsMetadata(new ApplicationSkills())).toEqual(
      getSkillsMetadata(ApplicationSkills),
    );
  });

  test('@Skill produces the same shape as agentSkill()', () => {
    @Skill({
      name: 'code-reviewer',
      description: 'Reviews TypeScript code.',
      content: 'Check nulls.',
    })
    class CodeReviewerSkill {}

    const declared = getDeclaredSkills(CodeReviewerSkill);
    expect(declared).toHaveLength(1);
    expect(declared[0]).toEqual(
      agentSkill({
        name: 'code-reviewer',
        description: 'Reviews TypeScript code.',
        content: 'Check nulls.',
      }),
    );
  });

  test('skillsToolboxOptionsFrom matches SkillsToolbox.builder().toOptions()', () => {
    @Skills({
      directories: ['.agents/skills'],
      packages: ['@company/skills'],
      workspace: '/tmp/app',
    })
    @SemanticSkillDiscovery({
      indexFile: '.di-framework/skills-index.jsonl',
      limit: 10,
    })
    @Skill({
      name: 'code-reviewer',
      description: 'Reviews TypeScript code.',
      content: 'Check nulls.',
    })
    class ApplicationSkills {}

    const fromDecorators = skillsToolboxOptionsFrom(ApplicationSkills);
    const fromBuilder = SkillsToolbox.builder()
      .addSkillsDirectories(['.agents/skills'])
      .addPackages(['@company/skills'])
      .workspace('/tmp/app')
      .addSkill(
        agentSkill({
          name: 'code-reviewer',
          description: 'Reviews TypeScript code.',
          content: 'Check nulls.',
        }),
      )
      .semanticDiscovery({
        indexFile: '.di-framework/skills-index.jsonl',
        limit: 10,
      })
      .toOptions();

    expect(fromDecorators).toEqual(fromBuilder);
    expect(skillsToolboxBuilderFrom(ApplicationSkills).toOptions()).toEqual(fromBuilder);
  });

  test('apply helpers accept embedder and chatModel overrides without storing them on metadata', () => {
    @Skills({ directories: ['.agents/skills'], sourceMode: 'replace' })
    @SemanticSkillDiscovery({ limit: 5 })
    class ApplicationSkills {}

    const embedder = new FakeEmbedder();
    const chatModel = new ScriptedChatModel([]);

    const options = skillsToolboxOptionsFrom(ApplicationSkills, {
      chatModel,
      semanticDiscovery: { embedder, limit: 3 },
    });

    expect(getSemanticSkillDiscoveryMetadata(ApplicationSkills)).toEqual({ limit: 5 });
    expect(options.chatModel).toBe(chatModel);
    expect(options.semanticDiscovery).toEqual({ limit: 3, embedder });

    expect(
      skillsToolboxOptionsFrom(ApplicationSkills, { semanticDiscovery: false }).semanticDiscovery,
    ).toBe(false);

    const agentOptions = skillsAgentBuilderFrom(ApplicationSkills, {
      chatModel,
      system: 'Be concise.',
    }).toAgentOptions();
    expect(agentOptions.chatModel).toBe(chatModel);
    expect(agentOptions.system).toBe('Be concise.');
    expect(agentOptions.semanticDiscovery).toEqual({ limit: 5 });
  });

  test('skillsIndexBuilderFrom matches SkillsIndex.builder().toOptions()', () => {
    @SkillsIndexConfig({
      directories: ['.agents/skills'],
      threshold: 50,
      retrievalLimit: 10,
      outputFile: '.di-framework/skills-index.jsonl',
    })
    class ApplicationSkillsIndex {}

    expect(getSkillsIndexMetadata(ApplicationSkillsIndex)).toEqual({
      directories: ['.agents/skills'],
      threshold: 50,
      retrievalLimit: 10,
      outputFile: '.di-framework/skills-index.jsonl',
    });

    const fromDecorators = skillsIndexBuilderFrom(ApplicationSkillsIndex).toOptions();
    const fromBuilder = SkillsIndex.builder()
      .addSkillsDirectories(['.agents/skills'])
      .threshold(50)
      .retrievalLimit(10)
      .outputFile('.di-framework/skills-index.jsonl')
      .toOptions();

    expect(fromDecorators).toEqual(fromBuilder);
  });

  test('skillsAgentBuilderFrom prefilled options match SkillsAgent.builder()', () => {
    @Skills({ directories: ['skills'], sourceMode: 'replace' })
    class ApplicationSkills {}

    const model = new ScriptedChatModel([]);
    const decorated = skillsAgentBuilderFrom(ApplicationSkills, {
      chatModel: model,
      write: true,
    }).toAgentOptions();
    const handBuilt = SkillsAgent.builder()
      .addSkillsDirectories(['skills'])
      .sourceMode('replace')
      .chatModel(model)
      .write()
      .toAgentOptions();

    expect(decorated).toEqual(handBuilt);
  });

  test('replace mode is preserved without explicit directories', () => {
    @Skills({ sourceMode: 'replace' })
    class EmptyCatalog {}

    expect(skillsToolboxOptionsFrom(EmptyCatalog).directories).toBeUndefined();
    expect(skillsToolboxBuilderFrom(EmptyCatalog).toOptions().sourceMode).toBe('replace');
  });

  test('skillsToolboxFrom and skillsAgentFrom build through existing factories', () => {
    @Skills({
      skills: [
        agentSkill({
          name: 'inline',
          description: 'Inline programmatic skill for decorator apply helpers.',
          content: 'Do the thing.',
        }),
      ],
      sourceMode: 'replace',
    })
    class ApplicationSkills {}

    const model = new ScriptedChatModel([]);
    const toolbox = skillsToolboxFrom(ApplicationSkills, { chatModel: model });
    expect(toolbox.skills.map((skill) => skill.name)).toEqual(['inline']);

    const agent = skillsAgentFrom(ApplicationSkills, { chatModel: model });
    expect(agent).toBeDefined();
  });

  test('skillsAgentBuilderFrom applies agent-only overrides', () => {
    @Skills({
      skills: [
        agentSkill({
          name: 'inline',
          description: 'Inline programmatic skill for decorator apply helpers.',
          content: 'Do the thing.',
        }),
      ],
      sourceMode: 'replace',
    })
    class ApplicationSkills {}

    const model = new ScriptedChatModel([]);
    const memory = MessageWindowChatMemory.of();
    const client = ChatClient.builder(model).build();
    const builderOptions = {};
    const options = skillsAgentBuilderFrom(ApplicationSkills, {
      chatModel: model,
      chatClient: client,
      system: 'Hello',
      extraTools: [],
      advisors: [passthroughAdvisor],
      conversationMemory: memory,
      defaultConversationId: 'c1',
      builder: builderOptions,
      instructionDiscovery: false,
    }).toAgentOptions();

    expect(options.system).toBe('Hello');
    expect(options.chatClient).toBe(client);
    expect(options.advisors).toEqual([passthroughAdvisor]);
    expect(options.conversationMemory).toBe(memory);
    expect(options.defaultConversationId).toBe('c1');
    expect(options.builder).toBe(builderOptions);
    expect(options.instructionDiscovery).toBe(false);
  });

  test('builderFrom applies files and extraAllowedDirectories', () => {
    @Skills({ sourceMode: 'replace' })
    class ApplicationSkills {}

    const options = skillsToolboxBuilderFrom(ApplicationSkills, {
      files: ['/tmp/does-not-need-to-exist/SKILL.md'],
      extraAllowedDirectories: ['/tmp/extra'],
      toolDescriptionTemplate: 'Skills:\n%s',
      skills: [
        agentSkill({
          name: 'inline',
          description: 'Inline programmatic skill for decorator apply helpers.',
          content: 'Do the thing.',
        }),
      ],
    }).toOptions();

    expect(options.files).toEqual(['/tmp/does-not-need-to-exist/SKILL.md']);
    expect(options.extraAllowedDirectories).toEqual(['/tmp/extra']);
    expect(options.toolDescriptionTemplate).toBe('Skills:\n%s');
  });
});
