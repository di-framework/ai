import { describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  ChatAgent,
  ChatClient,
  ScriptedChatModel,
  toolCall,
  toolCallResponse,
} from '@di-framework/ai';
import { agentSkill, SkillsTool } from '../src/index.ts';

describe('ChatAgent + SkillsTool', () => {
  test('agent loads a SKILL.md when the model invokes Skill', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-agent-'));
    const skillDir = join(root, 'code-reviewer');
    mkdirSync(skillDir);
    writeFileSync(
      join(skillDir, 'SKILL.md'),
      `---
name: code-reviewer
description: Reviews TypeScript for nulls and Spring-style conventions. Use when the user asks to review or audit code.
---

# Code Reviewer

1. Check for null pointer risks
2. Suggest a concrete fix
`,
    );

    const model = new ScriptedChatModel([
      {
        respond: toolCallResponse([toolCall('c1', 'Skill', { command: 'code-reviewer' })]),
      },
      {
        respond: (prompt) => {
          const blob = JSON.stringify(prompt.messages);
          expect(blob).toContain('Base directory for this skill:');
          expect(blob).toContain('# Code Reviewer');
          expect(blob).toContain(skillDir);
          return 'Review complete: watch for nulls.';
        },
      },
    ]);

    const agent = ChatAgent.create({
      chatModel: model,
      system: 'You help with code review.',
      tools: [SkillsTool.builder().addSkillsDirectory(root).build()],
    });

    const { content } = await agent.chat('Review this controller for best practices.');
    expect(content).toBe('Review complete: watch for nulls.');
  });

  test('ChatClient.defaultTools also hosts the Skill callback', async () => {
    const model = new ScriptedChatModel([
      { respond: toolCallResponse([toolCall('c1', 'Skill', { command: 'missing' })]) },
      {
        respond: (prompt) => {
          const blob = JSON.stringify(prompt.messages);
          expect(blob).toContain('Skill not found: missing');
          return 'no such skill';
        },
      },
    ]);

    const client = ChatClient.builder(model)
      .defaultTools(
        SkillsTool.of({
          skills: [agentSkill({ name: 'only', content: 'only this' })],
        }),
      )
      .build();

    const text = await client.prompt().user('use missing').call().content();
    expect(text).toBe('no such skill');
  });
});
