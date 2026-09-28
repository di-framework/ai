/**
 * Immutable context map passed into tool callbacks.
 * Spring AI: {@code ToolContext}.
 */
export class ToolContext {
  private readonly map: ReadonlyMap<string, unknown>;

  constructor(context: ReadonlyMap<string, unknown> | Readonly<Record<string, unknown>> = {}) {
    this.map = context instanceof Map ? new Map(context) : new Map(Object.entries(context));
  }

  get context(): ReadonlyMap<string, unknown> {
    return this.map;
  }

  get(key: string): unknown {
    return this.map.get(key);
  }

  has(key: string): boolean {
    return this.map.has(key);
  }

  isEmpty(): boolean {
    return this.map.size === 0;
  }

  toRecord(): Record<string, unknown> {
    return Object.fromEntries(this.map);
  }
}
