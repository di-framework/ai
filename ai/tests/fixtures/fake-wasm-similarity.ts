export function cosine_similarity_dataspace(
  dataspace: Float64Array,
  rows: number,
  dimensions: number,
  query: Float64Array,
): number[] {
  const pairs: number[] = [];
  for (let row = 0; row < rows; row++) {
    let dot = 0;
    let queryNorm = 0;
    let rowNorm = 0;
    for (let dimension = 0; dimension < dimensions; dimension++) {
      const a = query[dimension] ?? 0;
      const b = dataspace[row * dimensions + dimension] ?? 0;
      dot += a * b;
      queryNorm += a * a;
      rowNorm += b * b;
    }
    pairs.push(queryNorm && rowNorm ? dot / Math.sqrt(queryNorm * rowNorm) : 0, row);
  }
  return pairs;
}
