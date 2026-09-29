import { expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

test('ai workspace contains ai, ai-utils, and ml infer', () => {
  expect(existsSync('packages/ai/package.json')).toBe(true);
  expect(existsSync('packages/ai-utils/package.json')).toBe(true);
  expect(existsSync('packages/ml/infer/package.json')).toBe(true);
});
