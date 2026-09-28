import { describe, expect, test } from 'bun:test';
import { assistantMessage, Prompt, systemMessage, userMessage } from '../src/index.ts';

describe('Prompt', () => {
  test('constructs from string as user message', () => {
    const prompt = new Prompt('Hello');
    expect(prompt.messages).toHaveLength(1);
    expect(prompt.messages[0]?.messageType).toBe('user');
    expect(prompt.messages[0]?.text).toBe('Hello');
  });

  test('exposes instructions alias', () => {
    const prompt = Prompt.fromMessages([systemMessage('sys'), userMessage('hi')]);
    expect(prompt.instructions).toHaveLength(2);
    expect(prompt.getSystemMessage().text).toBe('sys');
    expect(prompt.getUserMessage().text).toBe('hi');
  });

  test('augmentSystemMessage inserts when missing', () => {
    const prompt = new Prompt('hi').augmentSystemMessage('You are helpful.');
    expect(prompt.messages[0]?.messageType).toBe('system');
    expect(prompt.messages[0]?.text).toBe('You are helpful.');
    expect(prompt.messages[1]?.text).toBe('hi');
  });

  test('augmentUserMessage replaces last user message', () => {
    const prompt = Prompt.fromMessages([
      systemMessage('s'),
      userMessage('old'),
      assistantMessage('a'),
    ]).augmentUserMessage('new');
    expect(prompt.getUserMessage().text).toBe('new');
  });

  test('getSystemMessages / getUserMessages return all matching messages', () => {
    const prompt = Prompt.fromMessages([
      systemMessage('s1'),
      userMessage('u1'),
      assistantMessage('a1'),
      userMessage('u2'),
    ]);
    expect(prompt.getSystemMessages().map((m) => m.text)).toEqual(['s1']);
    expect(prompt.getUserMessages().map((m) => m.text)).toEqual(['u1', 'u2']);
  });

  test('copy() clones messages/options into a new independent Prompt', () => {
    const original = new Prompt('hi', { temperature: 0.3 });
    const copy = original.copy();
    expect(copy.messages).toEqual(original.messages);
    expect(copy.messages).not.toBe(original.messages);
    expect(copy.options).toEqual(original.options);
    expect(copy.options).not.toBe(original.options);
  });

  test('copy() with no options stays undefined', () => {
    const copy = new Prompt('hi').copy();
    expect(copy.options).toBeUndefined();
  });

  test('Prompt.of() builds a user-message prompt with options', () => {
    const prompt = Prompt.of('hello', { temperature: 0.4 });
    expect(prompt.getUserMessage().text).toBe('hello');
    expect(prompt.options?.temperature).toBe(0.4);
  });

  test('withOptions merges provider options', () => {
    const prompt = new Prompt('x', {
      temperature: 0.2,
      providerOptions: { a: 1 },
    }).withOptions({
      maxTokens: 10,
      providerOptions: { b: 2 },
    });
    expect(prompt.options?.temperature).toBe(0.2);
    expect(prompt.options?.maxTokens).toBe(10);
    expect(prompt.options?.providerOptions).toEqual({ a: 1, b: 2 });
  });
});
