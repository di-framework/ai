import {
  type SkillAdapterCapabilities,
  SkillAdapterError,
  type SkillIndexWriteReceipt,
  type SkillIndexWriteRequest,
  type SkillIndexWriter,
} from './skill-adapters.ts';
import { type SkillSearchConnection, toChunkRecords } from './skill-search-connection.ts';

const CAPABILITIES: SkillAdapterCapabilities = {
  namespaces: true,
  lazyBodies: false,
  vectorSearch: false,
  indexWriting: true,
  eventuallyConsistent: false,
};

export class SkillSearchIndexer implements SkillIndexWriter {
  readonly capabilities = CAPABILITIES;
  private readonly connection: SkillSearchConnection;
  private readonly data?: SkillIndexWriteRequest;

  constructor(connection: SkillSearchConnection);
  constructor(data: SkillIndexWriteRequest, connection: SkillSearchConnection);
  constructor(
    dataOrConnection: SkillIndexWriteRequest | SkillSearchConnection,
    connection?: SkillSearchConnection,
  ) {
    if (connection) {
      this.data = dataOrConnection as SkillIndexWriteRequest;
      this.connection = connection;
    } else {
      this.connection = dataOrConnection as SkillSearchConnection;
    }
  }

  async replace(request?: SkillIndexWriteRequest): Promise<SkillIndexWriteReceipt> {
    return this.write('replace', request);
  }

  async upsert(request?: SkillIndexWriteRequest): Promise<SkillIndexWriteReceipt> {
    return this.write('upsert', request);
  }

  private async write(
    mode: 'replace' | 'upsert',
    request?: SkillIndexWriteRequest,
  ): Promise<SkillIndexWriteReceipt> {
    const data = request ?? this.data;
    if (!data) {
      throw new SkillAdapterError('INVALID_RESPONSE', 'Index write data is required');
    }
    try {
      const records = toChunkRecords(data);
      await this.connection.saveMetadata({ ...data.metadata, ready: false });
      if (mode === 'replace') {
        await this.connection.replaceChunks(data.metadata.namespace, records);
      } else {
        await this.connection.upsertChunks(records);
      }
      const metadata = { ...data.metadata, ready: true as const };
      await this.connection.saveMetadata(metadata);
      return { ...metadata, writtenVectors: records.length };
    } catch (error) {
      if (error instanceof SkillAdapterError) throw error;
      throw new SkillAdapterError('INVALID_RESPONSE', 'Skill index write failed', { cause: error });
    }
  }
}
