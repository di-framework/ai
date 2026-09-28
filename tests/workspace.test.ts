import { expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

test('ai workspace contains ai and ai-utils', () => {
  expect(existsSync('ai/package.json')).toBe(true);
  expect(existsSync('ai-utils/package.json')).toBe(true);
});
