export interface WasmSimilarity {
  cosine_similarity_dataspace(
    dataspace: Float64Array,
    rows: number,
    dimensions: number,
    query: Float64Array,
  ): ArrayLike<number>;
}

export async function loadWasmSimilarity(
  specifier = 'wasm-similarity',
): Promise<WasmSimilarity | null> {
  try {
    const module: unknown = await import(specifier);
    return typeof (module as WasmSimilarity | null)?.cosine_similarity_dataspace === 'function'
      ? (module as WasmSimilarity)
      : null;
  } catch {
    return null;
  }
}
