import { describe, expect, test } from 'bun:test';
import { parseSkillMarkdown, parseYaml } from '../src/index.ts';

describe('parseYaml', () => {
  test('parses maps, lists, and nested objects', () => {
    const value = parseYaml(`
name: code-reviewer
allowed-tools:
  - Read
  - Grep
metadata:
  author: ada
  tags:
    - review
    - ts
`);
    expect(value).toEqual({
      name: 'code-reviewer',
      'allowed-tools': ['Read', 'Grep'],
      metadata: { author: 'ada', tags: ['review', 'ts'] },
    });
  });

  test('parses block scalars', () => {
    const value = parseYaml(`
description: |
  line one
  line two
`);
    expect(value).toEqual({ description: 'line one\nline two' });
  });
});

describe('parseSkillMarkdown yaml', () => {
  test('reads list allowed-tools, license, and metadata', () => {
    const skill = parseSkillMarkdown(`---
name: code-reviewer
description: Reviews TypeScript when asked to review.
license: Apache-2.0
compatibility: Requires bun
allowed-tools:
  - Read
  - Grep
metadata:
  author: ada
---

# body
`);
    expect(skill.allowedTools).toEqual(['Read', 'Grep']);
    expect(skill.license).toBe('Apache-2.0');
    expect(skill.compatibility).toBe('Requires bun');
    expect(skill.metadata).toEqual({ author: 'ada' });
    expect(skill.content).toContain('# body');
  });
});
