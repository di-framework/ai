export interface HnswIndexOptions {
  readonly M?: number;
  readonly efConstruction?: number;
  readonly efSearch?: number;
}

export interface HnswHit {
  readonly id: string;
  readonly score: number;
}

export interface HnswSnapshotNode {
  readonly id: string;
  readonly level: number;
  readonly neighbors: readonly (readonly string[])[];
}

export interface HnswSnapshot {
  readonly M: number;
  readonly efConstruction: number;
  readonly efSearch: number;
  readonly entryPoint: string | null;
  readonly nodes: readonly HnswSnapshotNode[];
}

interface HnswNode {
  id: string;
  embedding: Float32Array;
  level: number;
  neighbors: string[][];
}

const DEFAULT_M = 16;
const DEFAULT_EF_CONSTRUCTION = 200;
const DEFAULT_EF_SEARCH = 64;

export class HnswIndex {
  readonly M: number;
  readonly Mmax0: number;
  readonly efConstruction: number;
  readonly efSearch: number;
  readonly mL: number;
  private entryPoint: string | null = null;
  private readonly nodes = new Map<string, HnswNode>();

  constructor(options: HnswIndexOptions = {}) {
    this.M = positiveInteger(options.M ?? DEFAULT_M, 'M');
    this.Mmax0 = this.M * 2;
    this.efConstruction = positiveInteger(
      options.efConstruction ?? DEFAULT_EF_CONSTRUCTION,
      'efConstruction',
    );
    this.efSearch = positiveInteger(options.efSearch ?? DEFAULT_EF_SEARCH, 'efSearch');
    this.mL = 1 / Math.log(this.M);
  }

  get size(): number {
    return this.nodes.size;
  }

  insert(id: string, embedding: ArrayLike<number>): void {
    const vector = Float32Array.from(embedding);
    if (this.nodes.has(id)) this.remove(id);
    const level = assignLevel(id, this.mL);
    const node: HnswNode = {
      id,
      embedding: vector,
      level,
      neighbors: Array.from({ length: level + 1 }, () => []),
    };
    this.nodes.set(id, node);
    const entryPoint = this.entryPoint;
    const entry = entryPoint != null ? this.nodes.get(entryPoint) : undefined;
    if (entryPoint == null || !entry) {
      this.entryPoint = id;
      return;
    }
    let current = entryPoint;
    for (let lc = entry.level; lc > level; lc--) {
      current = this.greedy(vector, current, lc);
    }
    for (let lc = Math.min(entry.level, level); lc >= 0; lc--) {
      const nearest = this.searchLayer(vector, [current], this.efConstruction, lc);
      const selected = this.selectNeighbors(
        vector,
        nearest.map((hit) => hit.id).filter((candidate) => candidate !== id),
        lc === 0 ? this.Mmax0 : this.M,
      );
      node.neighbors[lc] = selected;
      for (const neighborId of selected) {
        const neighbor = this.nodes.get(neighborId);
        if (!neighbor) continue;
        while (neighbor.neighbors.length <= lc) neighbor.neighbors.push([]);
        const links = neighbor.neighbors[lc] ?? [];
        if (!links.includes(id)) links.push(id);
        const maxLinks = lc === 0 ? this.Mmax0 : this.M;
        neighbor.neighbors[lc] =
          links.length > maxLinks
            ? this.selectNeighbors(neighbor.embedding, links, maxLinks)
            : links;
      }
      current = nearest[0]?.id ?? current;
    }
    if (level > entry.level) this.entryPoint = id;
  }

  search(query: ArrayLike<number>, k: number, ef = this.efSearch): HnswHit[] {
    if (k < 1 || this.entryPoint == null) return [];
    const vector = Float32Array.from(query);
    const entry = this.nodes.get(this.entryPoint);
    if (!entry) return [];
    let current = this.entryPoint;
    for (let lc = entry.level; lc > 0; lc--) {
      current = this.greedy(vector, current, lc);
    }
    const nearest = this.searchLayer(vector, [current], Math.max(ef, k), 0);
    return nearest.slice(0, k).map((hit) => ({ id: hit.id, score: 1 - hit.distance }));
  }

  remove(id: string): void {
    const node = this.nodes.get(id);
    if (!node) return;
    for (let lc = 0; lc <= node.level; lc++) {
      for (const neighborId of node.neighbors[lc] ?? []) {
        const neighbor = this.nodes.get(neighborId);
        const neighborLinks = neighbor?.neighbors[lc];
        if (!neighborLinks) continue;
        neighbor.neighbors[lc] = neighborLinks.filter((value) => value !== id);
      }
    }
    this.nodes.delete(id);
    if (this.entryPoint === id) {
      this.entryPoint = this.highestLevelNode();
    }
  }

  snapshot(): HnswSnapshot {
    return {
      M: this.M,
      efConstruction: this.efConstruction,
      efSearch: this.efSearch,
      entryPoint: this.entryPoint,
      nodes: [...this.nodes.values()].map((node) => ({
        id: node.id,
        level: node.level,
        neighbors: node.neighbors.map((links) => [...links]),
      })),
    };
  }

  static restore(
    snapshot: HnswSnapshot,
    embeddings: ReadonlyMap<string, Float32Array>,
    options?: HnswIndexOptions,
  ): HnswIndex {
    const index = new HnswIndex({
      M: options?.M ?? snapshot.M,
      efConstruction: options?.efConstruction ?? snapshot.efConstruction,
      efSearch: options?.efSearch ?? snapshot.efSearch,
    });
    for (const node of snapshot.nodes) {
      const embedding = embeddings.get(node.id);
      if (!embedding) continue;
      index.nodes.set(node.id, {
        id: node.id,
        embedding,
        level: node.level,
        neighbors: node.neighbors.map((links) => [...links]),
      });
    }
    index.entryPoint =
      snapshot.entryPoint && index.nodes.has(snapshot.entryPoint)
        ? snapshot.entryPoint
        : index.highestLevelNode();
    return index;
  }

  private greedy(query: Float32Array, entry: string, level: number): string {
    let current = entry;
    let changed = true;
    while (changed) {
      changed = false;
      const node = this.nodes.get(current);
      if (!node) return current;
      let best = current;
      let bestDistance = this.distance(query, node.embedding);
      for (const neighborId of node.neighbors[level] ?? []) {
        const neighbor = this.nodes.get(neighborId);
        if (!neighbor) continue;
        const distance = this.distance(query, neighbor.embedding);
        if (distance < bestDistance) {
          best = neighborId;
          bestDistance = distance;
          changed = true;
        }
      }
      current = best;
    }
    return current;
  }

  private searchLayer(
    query: Float32Array,
    entryPoints: readonly string[],
    ef: number,
    level: number,
  ): { id: string; distance: number }[] {
    const visited = new Set<string>();
    const candidates = new DistanceHeap('min');
    const found = new DistanceHeap('max');
    for (const id of entryPoints) {
      const node = this.nodes.get(id);
      if (!node || visited.has(id)) continue;
      const distance = this.distance(query, node.embedding);
      visited.add(id);
      candidates.push({ id, distance });
      found.push({ id, distance });
    }
    while (candidates.size > 0) {
      const closest = candidates.pop();
      if (!closest) break;
      const furthest = found.peek();
      if (furthest && closest.distance > furthest.distance && found.size >= ef) break;
      const node = this.nodes.get(closest.id);
      if (!node) continue;
      for (const neighborId of node.neighbors[level] ?? []) {
        if (visited.has(neighborId)) continue;
        const neighbor = this.nodes.get(neighborId);
        if (!neighbor) continue;
        visited.add(neighborId);
        const distance = this.distance(query, neighbor.embedding);
        const foundFurthest = found.peek();
        if (found.size < ef || !foundFurthest || distance < foundFurthest.distance) {
          candidates.push({ id: neighborId, distance });
          found.push({ id: neighborId, distance });
          if (found.size > ef) found.pop();
        }
      }
    }
    return found.toArray().sort((left, right) => left.distance - right.distance);
  }

  private selectNeighbors(query: Float32Array, candidates: readonly string[], M: number): string[] {
    const ranked = candidates
      .map((id) => {
        const node = this.nodes.get(id);
        return node
          ? { id, distance: this.distance(query, node.embedding), embedding: node.embedding }
          : null;
      })
      .filter(
        (value): value is { id: string; distance: number; embedding: Float32Array } =>
          value != null,
      )
      .sort((left, right) => left.distance - right.distance);
    const selected: typeof ranked = [];
    for (const candidate of ranked) {
      if (selected.length >= M) break;
      const closerToSelected = selected.some(
        (item) => this.distance(candidate.embedding, item.embedding) < candidate.distance,
      );
      if (!closerToSelected) selected.push(candidate);
    }
    if (selected.length < M) {
      for (const candidate of ranked) {
        if (selected.length >= M) break;
        if (!selected.some((item) => item.id === candidate.id)) selected.push(candidate);
      }
    }
    return selected.map((item) => item.id);
  }

  private distance(left: ArrayLike<number>, right: ArrayLike<number>): number {
    return 1 - cosine(left, right);
  }

  private highestLevelNode(): string | null {
    let best: HnswNode | undefined;
    for (const node of this.nodes.values()) {
      if (!best || node.level > best.level) best = node;
    }
    return best?.id ?? null;
  }
}

export function assignLevel(id: string, mL: number): number {
  const u = hash01(id);
  return Math.min(16, Math.floor(-Math.log(u) * mL));
}

export function cosine(left: ArrayLike<number>, right: ArrayLike<number>): number {
  if (left.length !== right.length) {
    throw new Error(`Embedding dimension mismatch: ${left.length} != ${right.length}`);
  }
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index++) {
    const a = left[index] ?? 0;
    const b = right[index] ?? 0;
    dot += a * b;
    leftNorm += a * a;
    rightNorm += b * b;
  }
  return leftNorm && rightNorm ? dot / Math.sqrt(leftNorm * rightNorm) : 0;
}

class DistanceHeap {
  private readonly items: { id: string; distance: number }[] = [];

  constructor(private readonly mode: 'min' | 'max') {}

  get size(): number {
    return this.items.length;
  }

  peek(): { id: string; distance: number } | undefined {
    return this.items[0];
  }

  push(item: { id: string; distance: number }): void {
    this.items.push(item);
    this.siftUp(this.items.length - 1);
  }

  pop(): { id: string; distance: number } | undefined {
    const top = this.items[0];
    const last = this.items.pop();
    if (this.items.length > 0 && last) {
      this.items[0] = last;
      this.siftDown(0);
    }
    return top;
  }

  toArray(): { id: string; distance: number }[] {
    return this.items.slice();
  }

  private better(left: { distance: number }, right: { distance: number }): boolean {
    return this.mode === 'min' ? left.distance < right.distance : left.distance > right.distance;
  }

  private siftUp(index: number): void {
    let current = index;
    while (current > 0) {
      const parent = (current - 1) >> 1;
      const node = this.items[current];
      const parentNode = this.items[parent];
      if (!node || !parentNode || !this.better(node, parentNode)) break;
      this.items[current] = parentNode;
      this.items[parent] = node;
      current = parent;
    }
  }

  private siftDown(index: number): void {
    let current = index;
    for (;;) {
      const left = current * 2 + 1;
      const right = left + 1;
      let best = current;
      const leftNode = this.items[left];
      const currentNode = this.items[best];
      if (
        left < this.items.length &&
        leftNode &&
        currentNode &&
        this.better(leftNode, currentNode)
      ) {
        best = left;
      }
      const rightNode = this.items[right];
      const bestNode = this.items[best];
      if (right < this.items.length && rightNode && bestNode && this.better(rightNode, bestNode)) {
        best = right;
      }
      if (best === current) break;
      const node = this.items[current];
      const swap = this.items[best];
      if (!node || !swap) break;
      this.items[current] = swap;
      this.items[best] = node;
      current = best;
    }
  }
}

function hash01(id: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < id.length; index++) {
    hash ^= id.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return ((hash >>> 0) + 0.5) / 4294967296;
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) throw new Error(`${name} must be a positive integer`);
  return value;
}
