import { createHash } from 'node:crypto';
import type { AgentSkill } from './parse-skill-markdown.ts';
import type {
  SkillAdapterCapabilities,
  SkillAdapterHealth,
  SkillCatalogListOptions,
  SkillCatalogLoadOptions,
  SkillCatalogStore,
  SkillDescriptor,
} from './skill-adapters.ts';
import { SkillAdapterError } from './skill-adapters.ts';
import { hashSkillCatalog } from './skills-index.ts';
import { collectSkills, type SkillsToolOptions } from './skills-tool.ts';

const CAPABILITIES: SkillAdapterCapabilities = {
  namespaces: false,
  lazyBodies: false,
  vectorSearch: false,
  indexWriting: false,
  eventuallyConsistent: false,
};

/** Existing filesystem discovery exposed through the catalog contract. */
export class LocalSkillCatalogStore implements SkillCatalogStore {
  readonly capabilities = CAPABILITIES;
  private readonly skills: readonly AgentSkill[];
  private readonly skillsByName: ReadonlyMap<string, AgentSkill>;
  private readonly descriptors: readonly SkillDescriptor[];
  private readonly catalogVersion: string;

  constructor(options: SkillsToolOptions | readonly AgentSkill[]) {
    this.skills = Array.isArray(options)
      ? (options as readonly AgentSkill[]).slice()
      : collectSkills(options as SkillsToolOptions);
    this.descriptors = this.skills
      .map((skill) => ({
        name: skill.name,
        description: skill.description,
        sourceHash: hash(skill.source),
        version: hash(skill.source),
      }))
      .sort((left, right) => left.name.localeCompare(right.name));
    this.skillsByName = new Map(this.skills.map((skill) => [skill.name, skill]));
    this.catalogVersion = hashSkillCatalog(this.skills);
  }

  list(options: SkillCatalogListOptions = {}): Promise<readonly SkillDescriptor[]> {
    const error = namespaceError(options.namespace);
    if (error) return Promise.reject(error);
    return Promise.resolve(this.descriptors);
  }

  load(name: string, options: SkillCatalogLoadOptions = {}): Promise<AgentSkill | undefined> {
    const error = namespaceError(options.namespace);
    if (error) return Promise.reject(error);
    const descriptor = this.descriptors.find((candidate) => candidate.name === name);
    if (!descriptor) return Promise.resolve(undefined);
    if (options.expectedVersion != null && descriptor?.version !== options.expectedVersion) {
      return Promise.reject(
        new SkillAdapterError('STALE_CATALOG', `Skill '${name}' changed before activation`),
      );
    }
    return Promise.resolve(this.skillsByName.get(name));
  }

  version(options: SkillCatalogListOptions = {}): Promise<string> {
    const error = namespaceError(options.namespace);
    if (error) return Promise.reject(error);
    return Promise.resolve(this.catalogVersion);
  }

  health(options: SkillCatalogListOptions = {}): Promise<SkillAdapterHealth> {
    const error = namespaceError(options.namespace);
    if (error) return Promise.reject(error);
    return Promise.resolve({ status: 'ready', checkedVersion: this.catalogVersion });
  }
}

function namespaceError(namespace?: string): Error | undefined {
  return namespace == null
    ? undefined
    : new Error('Local filesystem catalog does not support namespaces');
}

function hash(text: string): string {
  return createHash('sha256').update(text).digest('hex');
}
