import { AiError } from '../model/errors.ts';
import { A2AClient, type A2AFetchLike, type A2AHeadersInit } from './client.ts';
import type { AgentCard } from './types.ts';

export interface A2ADirectoryOptions {
  /** Initial list of remote Agent Card origin base URLs. */
  readonly origins?: readonly string[];
  /** Custom fetch implementation (defaults to global fetch). */
  readonly fetch?: A2AFetchLike;
  /** Custom headers (e.g. auth credentials) passed down to created clients. */
  readonly headers?: A2AHeadersInit | (() => A2AHeadersInit | Promise<A2AHeadersInit>);
}

export interface FindAgentOptions {
  /** The skill ID to discover (e.g. 'dev.review', 'math.calculate'). */
  readonly skill: string;
}

/**
 * Registry and discovery service for remote A2A 1.0 Agent origins.
 *
 * Discovers agents by querying registered `/.well-known/agent-card.json` endpoints
 * and returning connected HTTP `A2AClient` instances for remote execution.
 */
function trimTrailingSlashes(str: string): string {
  let end = str.length;
  while (end > 0 && str.charCodeAt(end - 1) === 47 /* '/' */) {
    end--;
  }
  return str.slice(0, end);
}

export class A2ADirectory {
  private readonly origins = new Set<string>();
  private readonly clientCache = new Map<string, A2AClient>();
  private readonly customFetch?: A2AFetchLike;
  private readonly headers?: A2AHeadersInit | (() => A2AHeadersInit | Promise<A2AHeadersInit>);

  constructor(options: A2ADirectoryOptions = {}) {
    this.customFetch = options.fetch;
    this.headers = options.headers;
    if (options.origins) {
      this.addAll(options.origins);
    }
  }

  static create(options: A2ADirectoryOptions = {}): A2ADirectory {
    return new A2ADirectory(options);
  }

  static of(...origins: string[]): A2ADirectory {
    return new A2ADirectory({ origins });
  }

  add(origin: string): this {
    const normalized = trimTrailingSlashes(origin);
    this.origins.add(normalized);
    return this;
  }

  addAll(origins: readonly string[]): this {
    for (const origin of origins) {
      this.add(origin);
    }
    return this;
  }

  remove(origin: string): boolean {
    const normalized = trimTrailingSlashes(origin);
    this.clientCache.delete(normalized);
    return this.origins.delete(normalized);
  }

  getOrigins(): readonly string[] {
    return Array.from(this.origins);
  }

  private getClient(origin: string): A2AClient {
    let client = this.clientCache.get(origin);
    if (!client) {
      client = A2AClient.create({
        baseUrl: origin,
        fetch: this.customFetch,
        headers: this.headers,
      });
      this.clientCache.set(origin, client);
    }
    return client;
  }

  /**
   * Discovers a remote agent that advertises the requested skill.
   *
   * Fetches Agent Cards from registered origins and returns an HTTP A2AClient
   * for the first matching peer. Fails closed with an AiError if not found.
   */
  async find(options: FindAgentOptions): Promise<A2AClient> {
    if (!options?.skill) {
      throw new AiError("Find options require a non-empty 'skill' ID", 'invalid-request');
    }

    const { skill } = options;

    for (const origin of this.origins) {
      const client = this.getClient(origin);
      let card: AgentCard;
      try {
        card = await client.getCard();
      } catch (_err) {
        // Skip unreachable origin or continue search
        continue;
      }

      const matches = card.skills?.some((s) => s.id === skill || s.name === skill);

      if (matches) {
        return client;
      }
    }

    throw new AiError(
      `No agent advertising skill '${skill}' found across ${this.origins.size} registered origins`,
      'invalid-request',
    );
  }

  /**
   * Finds all remote agents that advertise the requested skill (or all registered agents).
   */
  async findAll(options?: { skill?: string }): Promise<readonly A2AClient[]> {
    const matches: A2AClient[] = [];

    for (const origin of this.origins) {
      const client = this.getClient(origin);
      if (!options?.skill) {
        matches.push(client);
        continue;
      }

      try {
        const card = await client.getCard();
        if (card.skills?.some((s) => s.id === options.skill || s.name === options.skill)) {
          matches.push(client);
        }
      } catch {
        // Skip unreachable origins
      }
    }

    return matches;
  }

  clearCache(): void {
    this.clientCache.clear();
  }
}
