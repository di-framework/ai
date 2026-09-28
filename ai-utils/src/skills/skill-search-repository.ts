import {
  assertReadyMetadata,
  type SkillAdapterCapabilities,
  SkillAdapterError,
  type SkillAdapterHealth,
  type SkillCatalogListOptions,
  type SkillChunkMatch,
  type SkillVectorIndexMetadata,
  type SkillVectorQueryOptions,
  type SkillVectorSearch,
} from './skill-adapters.ts';
import type { SkillSearchConnection } from './skill-search-connection.ts';

const CAPABILITIES: SkillAdapterCapabilities = {
  namespaces: true,
  lazyBodies: false,
  vectorSearch: true,
  indexWriting: false,
  eventuallyConsistent: false,
};

export class SkillSearchRepository implements SkillVectorSearch {
  readonly capabilities = CAPABILITIES;

  constructor(private readonly connection: SkillSearchConnection) {}

  async metadata(options: SkillCatalogListOptions = {}): Promise<SkillVectorIndexMetadata> {
    return this.readyMetadata(options.namespace);
  }

  async health(options: SkillCatalogListOptions = {}): Promise<SkillAdapterHealth> {
    try {
      const metadata = await this.connection.loadMetadata(options.namespace);
      if (!metadata?.ready) {
        return { status: 'not-ready', message: 'Index is not ready' };
      }
      return { status: 'ready', checkedVersion: metadata.indexVersion };
    } catch (error) {
      throw wrap(error, 'Checking skill search health');
    }
  }

  async query(
    vector: ArrayLike<number>,
    options: SkillVectorQueryOptions = {},
  ): Promise<readonly SkillChunkMatch[]> {
    const metadata = await this.readyMetadata(options.namespace);
    assertReadyMetadata(metadata, options);
    if (vector.length !== metadata.dimensions) {
      throw new SkillAdapterError(
        'INVALID_RESPONSE',
        `Query embedding has ${vector.length} dimensions; expected ${metadata.dimensions}`,
      );
    }
    const limit = options.limit ?? 10;
    if (!Number.isInteger(limit) || limit < 1) {
      throw new SkillAdapterError('INVALID_RESPONSE', 'limit must be a positive integer');
    }
    const minScore = options.minScore ?? Number.NEGATIVE_INFINITY;
    if (Number.isNaN(minScore)) {
      throw new SkillAdapterError('INVALID_RESPONSE', 'minScore must be a number');
    }
    try {
      return await this.connection.queryByEmbedding({
        vector,
        limit,
        minScore,
        namespace: options.namespace,
      });
    } catch (error) {
      throw wrap(error, 'Searching skill vectors');
    }
  }

  private async readyMetadata(namespace?: string): Promise<SkillVectorIndexMetadata> {
    let metadata: SkillVectorIndexMetadata | undefined;
    try {
      metadata = await this.connection.loadMetadata(namespace);
    } catch (error) {
      throw wrap(error, 'Loading skill vector metadata');
    }
    if (!metadata?.ready) {
      throw new SkillAdapterError(
        'NOT_READY',
        `No ready skill index for namespace '${namespace ?? ''}'`,
      );
    }
    return metadata;
  }
}

function wrap(error: unknown, operation: string): SkillAdapterError {
  if (error instanceof SkillAdapterError) return error;
  return new SkillAdapterError('INVALID_RESPONSE', `${operation} failed`, { cause: error });
}
