import { mock } from 'bun:test';

// Runs before any test file loads (via bunfig.toml [test].preload), guaranteeing
// this registers before any real `import('wasm-similarity')` resolves. Bun's
// `mock.module` cannot retroactively override an already-resolved module, and
// `bun-sqlite.ts` memoizes its wasm import for the lifetime of the process, so
// ordering here is load-bearing for exercising the pure-JS `cosine` fallback.
mock.module('wasm-similarity', () => ({}));
