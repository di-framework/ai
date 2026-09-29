import { describe, expect, it } from 'bun:test';
import { AgentCardBuilder, AgentCardHelper } from '../../src/a2a/agent-card.ts';

describe('AgentCard Builder & Serializer', () => {
  it('builds an AgentCard via fluent builder with all options', () => {
    const card = AgentCardBuilder.create()
      .name('Aria')
      .description('Senior Code Reviewer')
      .version('1.2.0')
      .url('https://agents.example.com/aria', 'JSONRPC')
      .interface({
        url: 'https://agents.example.com/aria/custom',
        protocol_version: '1.0',
        protocol_binding: 'JSONRPC',
      })
      .skill({
        id: 'dev.review',
        name: 'Code Review',
        description: 'Review pull request diffs',
        tags: ['code', 'review'],
        examples: ['Review PR #42'],
        input_modes: ['text/plain'],
        output_modes: ['application/json'],
        security: [{ bearerAuth: [] }],
      })
      .skills({
        id: 'dev.refactor',
        description: 'Refactor code',
      })
      .capabilities({
        streaming: true,
        push_notifications: false,
        state_transition_history: true,
      })
      .defaultInputModes('text/plain', 'application/json')
      .defaultOutputModes('application/json')
      .securityScheme('bearerAuth', {
        type: 'http',
        scheme: 'bearer',
        bearerFormat: 'JWT',
      })
      .securityRequirement({ bearerAuth: [] })
      .provider({ organization: 'Example Org', url: 'https://example.com' })
      .documentationUrl('https://docs.example.com/aria')
      .iconUrl('https://example.com/icon.png')
      .build();

    expect(card.name).toBe('Aria');
    expect(card.description).toBe('Senior Code Reviewer');
    expect(card.version).toBe('1.2.0');
    expect(card.skills.length).toBe(2);
    expect(card.skills[0]?.tags).toEqual(['code', 'review']);
    expect(card.supported_interfaces.length).toBe(2);
    expect(card.capabilities?.streaming).toBe(true);
    expect(card.default_input_modes).toEqual(['text/plain', 'application/json']);
    expect(card.default_output_modes).toEqual(['application/json']);
    expect(card.security_schemes?.bearerAuth).toBeDefined();
    expect(card.security_requirements?.[0]?.bearerAuth).toBeDefined();
    expect(card.provider?.organization).toBe('Example Org');
    expect(card.documentation_url).toBe('https://docs.example.com/aria');
    expect(card.icon_url).toBe('https://example.com/icon.png');
  });

  it('creates an AgentCard from declarative options matching @Agent', () => {
    const card = AgentCardHelper.create({
      name: 'Ravi',
      description: 'QA Engineer',
      version: '2.0.0',
      skills: [
        {
          id: 'dev.test',
          description: 'Run automated tests',
        },
      ],
      a2a: {
        url: 'https://agents.example.com/ravi',
      },
    });

    expect(card.name).toBe('Ravi');
    expect(card.supported_interfaces[0]?.url).toBe('https://agents.example.com/ravi');
    expect(card.supported_interfaces[0]?.protocol_binding).toBe('JSONRPC');
  });

  it('round-trips JSON serialization without leaking private fields', () => {
    const card = AgentCardHelper.create({
      name: 'Forge',
      description: 'Task Manager',
      a2a: { url: 'https://agents.example.com/forge' },
    });

    const json = AgentCardHelper.serialize(card);
    const json2 = AgentCardHelper.toJson(card);
    expect(json).toContain('"name": "Forge"');
    expect(json2).toContain('"name": "Forge"');
    expect(json).not.toContain('prompt');
    expect(json).not.toContain('tool');
    expect(json).not.toContain('memory');

    const parsed = AgentCardHelper.deserialize(json);
    const parsed2 = AgentCardHelper.fromJson(json2);
    expect(parsed.name).toBe('Forge');
    expect(parsed2.name).toBe('Forge');
    expect(parsed.description).toBe('Task Manager');

    const fromBuilderHelper = AgentCardHelper.builder().name('Test').build();
    expect(fromBuilderHelper.name).toBe('Test');
  });

  it('rejects card with missing name', () => {
    expect(() => AgentCardBuilder.create().build()).toThrow();
  });
});
