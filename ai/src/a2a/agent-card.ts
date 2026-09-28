import { AiError } from '../model/errors.ts';
import {
  A2A_PROTOCOL_VERSION,
  type AgentCapabilities,
  type AgentCard,
  type AgentInterface,
  type AgentSkill,
  type SecurityRequirement,
  type SecurityScheme,
} from './types.ts';

export interface AgentCardSkillOptions {
  readonly id: string;
  readonly name?: string;
  readonly description: string;
  readonly tags?: readonly string[];
  readonly examples?: readonly string[];
  readonly input_modes?: readonly string[];
  readonly output_modes?: readonly string[];
  readonly security?: readonly Readonly<Record<string, readonly string[]>>[];
}

export interface AgentCardA2AOptions {
  readonly url: string;
  readonly binding?: 'JSONRPC' | 'HTTP';
  readonly protocolVersion?: string;
  readonly tenant?: string;
}

export interface AgentCardOptions {
  readonly name: string;
  readonly description?: string;
  readonly version?: string;
  readonly skills?: readonly (AgentSkill | AgentCardSkillOptions)[];
  readonly a2a?: AgentCardA2AOptions;
  readonly supported_interfaces?: readonly AgentInterface[];
  readonly capabilities?: AgentCapabilities;
  readonly default_input_modes?: readonly string[];
  readonly default_output_modes?: readonly string[];
  readonly security_schemes?: Readonly<Record<string, SecurityScheme>>;
  readonly security_requirements?: readonly SecurityRequirement[];
  readonly provider?: {
    readonly organization?: string;
    readonly url?: string;
  };
  readonly documentation_url?: string;
  readonly icon_url?: string;
}

/**
 * Fluent builder for creating compliant A2A 1.0 Agent Cards.
 */
export class AgentCardBuilder {
  static create(): AgentCardBuilder {
    return new AgentCardBuilder();
  }

  private _name?: string;
  private _description?: string;
  private _version?: string;
  private _interfaces: AgentInterface[] = [];
  private _skills: AgentSkill[] = [];
  private _capabilities: AgentCapabilities = {
    streaming: false,
    push_notifications: false,
    batch: false,
    state_transition_history: false,
  };
  private _default_input_modes?: string[];
  private _default_output_modes?: string[];
  private _security_schemes?: Record<string, SecurityScheme>;
  private _security_requirements?: SecurityRequirement[];
  private _provider?: { organization?: string; url?: string };
  private _documentation_url?: string;
  private _icon_url?: string;

  name(name: string): this {
    this._name = name;
    return this;
  }

  description(description: string): this {
    this._description = description;
    return this;
  }

  version(version: string): this {
    this._version = version;
    return this;
  }

  url(
    url: string,
    binding: 'JSONRPC' | 'HTTP' = 'JSONRPC',
    protocolVersion: string = A2A_PROTOCOL_VERSION,
  ): this {
    this._interfaces.push({
      url,
      protocol_version: protocolVersion,
      protocol_binding: binding,
    });
    return this;
  }

  interface(iface: AgentInterface): this {
    this._interfaces.push(iface);
    return this;
  }

  skill(skill: AgentSkill | AgentCardSkillOptions): this {
    this._skills.push({
      id: skill.id,
      ...(skill.name ? { name: skill.name } : {}),
      description: skill.description,
      ...(skill.tags ? { tags: [...skill.tags] } : {}),
      ...(skill.examples ? { examples: [...skill.examples] } : {}),
      ...(skill.input_modes ? { input_modes: [...skill.input_modes] } : {}),
      ...(skill.output_modes ? { output_modes: [...skill.output_modes] } : {}),
      ...(skill.security ? { security: [...skill.security] } : {}),
    });
    return this;
  }

  skills(...skills: readonly (AgentSkill | AgentCardSkillOptions)[]): this {
    for (const s of skills) {
      this.skill(s);
    }
    return this;
  }

  capabilities(caps: Partial<AgentCapabilities>): this {
    this._capabilities = {
      ...this._capabilities,
      ...caps,
    };
    return this;
  }

  defaultInputModes(...modes: string[]): this {
    this._default_input_modes = [...modes];
    return this;
  }

  defaultOutputModes(...modes: string[]): this {
    this._default_output_modes = [...modes];
    return this;
  }

  securityScheme(name: string, scheme: SecurityScheme): this {
    if (!this._security_schemes) {
      this._security_schemes = {};
    }
    this._security_schemes[name] = scheme;
    return this;
  }

  securityRequirement(req: SecurityRequirement): this {
    if (!this._security_requirements) {
      this._security_requirements = [];
    }
    this._security_requirements.push(req);
    return this;
  }

  provider(provider: { organization?: string; url?: string }): this {
    this._provider = provider;
    return this;
  }

  documentationUrl(url: string): this {
    this._documentation_url = url;
    return this;
  }

  iconUrl(url: string): this {
    this._icon_url = url;
    return this;
  }

  build(): AgentCard {
    if (!this._name) {
      throw new AiError('AgentCard requires a non-empty name', 'invalid-request');
    }

    const card: AgentCard = {
      name: this._name,
      ...(this._description ? { description: this._description } : {}),
      ...(this._version ? { version: this._version } : {}),
      supported_interfaces: [...this._interfaces],
      skills: [...this._skills],
      capabilities: this._capabilities,
      ...(this._default_input_modes ? { default_input_modes: this._default_input_modes } : {}),
      ...(this._default_output_modes ? { default_output_modes: this._default_output_modes } : {}),
      ...(this._security_schemes ? { security_schemes: this._security_schemes } : {}),
      ...(this._security_requirements
        ? { security_requirements: this._security_requirements }
        : {}),
      ...(this._provider ? { provider: this._provider } : {}),
      ...(this._documentation_url ? { documentation_url: this._documentation_url } : {}),
      ...(this._icon_url ? { icon_url: this._icon_url } : {}),
    };

    return card;
  }
}

/**
 * AgentCard utilities for creating, validating, and serializing A2A 1.0 Agent Cards.
 */
export const AgentCardHelper = {
  create(options: AgentCardOptions): AgentCard {
    return AgentCardHelper.fromOptions(options);
  },

  builder(): AgentCardBuilder {
    return new AgentCardBuilder();
  },

  fromOptions(options: AgentCardOptions): AgentCard {
    const builder = new AgentCardBuilder();
    builder.name(options.name);

    if (options.description) builder.description(options.description);
    if (options.version) builder.version(options.version);

    if (options.supported_interfaces) {
      for (const iface of options.supported_interfaces) {
        builder.interface(iface);
      }
    } else if (options.a2a) {
      builder.url(
        options.a2a.url,
        options.a2a.binding ?? 'JSONRPC',
        options.a2a.protocolVersion ?? A2A_PROTOCOL_VERSION,
      );
    }

    if (options.skills) {
      builder.skills(...options.skills);
    }

    if (options.capabilities) {
      builder.capabilities(options.capabilities);
    }

    if (options.default_input_modes) {
      builder.defaultInputModes(...options.default_input_modes);
    }

    if (options.default_output_modes) {
      builder.defaultOutputModes(...options.default_output_modes);
    }

    if (options.security_schemes) {
      for (const [name, scheme] of Object.entries(options.security_schemes)) {
        builder.securityScheme(name, scheme);
      }
    }

    if (options.security_requirements) {
      for (const req of options.security_requirements) {
        builder.securityRequirement(req);
      }
    }

    if (options.provider) {
      builder.provider(options.provider);
    }

    if (options.documentation_url) {
      builder.documentationUrl(options.documentation_url);
    }

    if (options.icon_url) {
      builder.iconUrl(options.icon_url);
    }

    return builder.build();
  },

  serialize(card: AgentCard, space = 2): string {
    return JSON.stringify(card, null, space);
  },

  toJson(card: AgentCard, space = 2): string {
    return AgentCardHelper.serialize(card, space);
  },

  parse(json: string | object): AgentCard {
    const obj = typeof json === 'string' ? JSON.parse(json) : json;
    const hasInterfaces =
      obj &&
      typeof obj === 'object' &&
      (Array.isArray((obj as Record<string, unknown>).supported_interfaces) ||
        Array.isArray((obj as Record<string, unknown>).interfaces));

    if (
      !obj ||
      typeof obj !== 'object' ||
      typeof (obj as Record<string, unknown>).name !== 'string' ||
      !hasInterfaces
    ) {
      throw new AiError('Invalid AgentCard JSON structure', 'invalid-request');
    }
    return obj as AgentCard;
  },

  deserialize(json: string | object): AgentCard {
    return AgentCardHelper.parse(json);
  },

  fromJson(json: string | object): AgentCard {
    return AgentCardHelper.parse(json);
  },
};
