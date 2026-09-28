import { describe, expect, test } from 'bun:test';
import {
  AiError,
  ChatClient,
  defaultResponseTextCleaner,
  FakeChatModel,
  listOutputConverter,
  mapOutputConverter,
  markdownCodeBlockCleaner,
  ScriptedChatModel,
  schemaOutputConverter,
  textResponse,
  thinkingTagCleaner,
  validateAgainstJsonSchema,
} from '../src/index.ts';

const personSchema = {
  type: 'object',
  properties: {
    name: { type: 'string' },
    age: { type: 'number' },
  },
  required: ['name', 'age'],
} as const;

describe('Response text cleaners', () => {
  test('markdownCodeBlockCleaner strips fenced json', () => {
    const raw = '```json\n{"a":1}\n```';
    expect(markdownCodeBlockCleaner(raw)).toBe('{"a":1}');
  });

  test('thinkingTagCleaner removes reasoning blocks', () => {
    const raw = '<thinking>plan</thinking>{"ok":true}';
    expect(thinkingTagCleaner(raw)).toBe('{"ok":true}');
  });

  test('default cleaner pipeline', () => {
    const raw = '  <think>x</think>\n```json\n{"n":2}\n```  ';
    expect(defaultResponseTextCleaner(raw)).toBe('{"n":2}');
  });

  test('markdownCodeBlockCleaner strips a single-line fenced block with no language tag', () => {
    expect(markdownCodeBlockCleaner('```{"a":1}```')).toBe('{"a":1}');
  });

  test('markdownCodeBlockCleaner strips a single-line fenced block with a language tag', () => {
    expect(markdownCodeBlockCleaner('```json {"a":1}```')).toBe('{"a":1}');
  });

  test('markdownCodeBlockCleaner keeps single-line content that looks like a language tag but is not JSON', () => {
    expect(markdownCodeBlockCleaner('```plain text```')).toBe('plain text');
  });
});

describe('SchemaOutputConverter', () => {
  test('parses clean JSON', () => {
    const converter = schemaOutputConverter<{ name: string; age: number }>({
      schema: personSchema,
    });
    expect(converter.convert('{"name":"Ada","age":36}')).toEqual({
      name: 'Ada',
      age: 36,
    });
  });

  test('parses fenced JSON', () => {
    const converter = schemaOutputConverter({ schema: personSchema });
    expect(converter.convert('```json\n{"name":"Bob","age":20}\n```')).toEqual({
      name: 'Bob',
      age: 20,
    });
  });

  test('throws AiError on invalid JSON', () => {
    const converter = schemaOutputConverter({ schema: personSchema });
    expect(() => converter.convert('not-json')).toThrow(AiError);
  });

  test('getFormat includes schema', () => {
    const converter = schemaOutputConverter({ schema: personSchema });
    expect(converter.getFormat()).toContain('JSON Schema');
    expect(converter.getJsonSchema()).toContain('name');
  });

  test('mapOutputConverter', () => {
    const converter = mapOutputConverter();
    expect(converter.convert('{"x":1}')).toEqual({ x: 1 });
  });

  test('listOutputConverter', () => {
    const converter = listOutputConverter<number>();
    expect(converter.convert('[1,2,3]')).toEqual([1, 2, 3]);
  });
});

describe('JSON schema validator', () => {
  test('accepts valid object', () => {
    const result = validateAgainstJsonSchema({ name: 'Ada', age: 36 }, personSchema);
    expect(result.success).toBe(true);
  });

  test('rejects missing required', () => {
    const result = validateAgainstJsonSchema({ name: 'Ada' }, personSchema);
    expect(result.success).toBe(false);
    expect(result.errorMessage).toContain('age');
  });

  test('rejects wrong type', () => {
    const result = validateAgainstJsonSchema({ name: 'Ada', age: 'old' }, personSchema);
    expect(result.success).toBe(false);
  });

  test('rejects excessive nesting depth', () => {
    let schema: Record<string, unknown> = { type: 'number' };
    let value: unknown = 1;
    for (let i = 0; i < 8; i++) {
      schema = { type: 'object', properties: { c: schema } };
      value = { c: value };
    }
    const result = validateAgainstJsonSchema(value, schema, 5);
    expect(result.success).toBe(false);
    expect(result.errorMessage).toContain('max validation depth');
  });

  test('accepts a schema passed as a JSON string', () => {
    const result = validateAgainstJsonSchema(
      { name: 'Ada', age: 36 },
      JSON.stringify(personSchema),
    );
    expect(result.success).toBe(true);
  });

  test('rejects an invalid JSON string schema', () => {
    const result = validateAgainstJsonSchema({ name: 'Ada' }, '{not json');
    expect(result.success).toBe(false);
    expect(result.errorMessage).toContain('Invalid JSON schema');
  });

  test('supports an array of allowed types', () => {
    const schema = { type: ['string', 'number'] };
    expect(validateAgainstJsonSchema('x', schema).success).toBe(true);
    expect(validateAgainstJsonSchema(1, schema).success).toBe(true);
    const bad = validateAgainstJsonSchema(true, schema);
    expect(bad.success).toBe(false);
    expect(bad.errorMessage).toContain('expected one of types');
  });

  test('validates const values', () => {
    const ok = validateAgainstJsonSchema('fixed', { const: 'fixed' });
    expect(ok.success).toBe(true);
    const bad = validateAgainstJsonSchema('other', { const: 'fixed' });
    expect(bad.success).toBe(false);
    expect(bad.errorMessage).toContain('expected const');
  });

  test('validates enum values', () => {
    const ok = validateAgainstJsonSchema('b', { enum: ['a', 'b', 'c'] });
    expect(ok.success).toBe(true);
    const bad = validateAgainstJsonSchema('z', { enum: ['a', 'b', 'c'] });
    expect(bad.success).toBe(false);
    expect(bad.errorMessage).toContain('not in enum');
  });

  test('validates array items against a schema', () => {
    const schema = { type: 'array', items: { type: 'number' } };
    expect(validateAgainstJsonSchema([1, 2, 3], schema).success).toBe(true);
    const bad = validateAgainstJsonSchema([1, 'x', 3], schema);
    expect(bad.success).toBe(false);
    expect(bad.errorMessage).toContain('$[1]');
  });

  test('non-array value against array type short-circuits', () => {
    const result = validateAgainstJsonSchema('not-an-array', { type: 'array' });
    expect(result.success).toBe(false);
  });

  test('matchesType covers integer/boolean/null/unknown branches', () => {
    expect(validateAgainstJsonSchema(3, { type: 'integer' }).success).toBe(true);
    expect(validateAgainstJsonSchema(3.5, { type: 'integer' }).success).toBe(false);
    expect(validateAgainstJsonSchema(true, { type: 'boolean' }).success).toBe(true);
    expect(validateAgainstJsonSchema(null, { type: 'null' }).success).toBe(true);
    expect(validateAgainstJsonSchema(NaN, { type: 'number' }).success).toBe(false);
    expect(validateAgainstJsonSchema('anything', { type: 'unknown-type' }).success).toBe(true);
  });

  test('object-typed schema short-circuits on non-object/array values', () => {
    // `type: 'object'` mismatch fails the type check itself before required-key
    // validation runs, so this exercises the early-return guard at that branch.
    const result = validateAgainstJsonSchema('x', { type: 'object', required: ['a'] });
    expect(result.success).toBe(false);
  });

  test('deepEqual falls back to false for values that cannot be stringified', () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    const result = validateAgainstJsonSchema(circular, { const: { other: true } });
    expect(result.success).toBe(false);
  });

  test('ignores properties without a matching schema entry', () => {
    const schema = { type: 'object', properties: { known: { type: 'string' } } };
    const result = validateAgainstJsonSchema({ known: 'ok', extra: 1 }, schema);
    expect(result.success).toBe(true);
  });
});

describe('ChatClient.entity', () => {
  test('entity with schemaOutputConverter', async () => {
    const model = new FakeChatModel('{"name":"Ada","age":36}');
    const converter = schemaOutputConverter<{ name: string; age: number }>({
      schema: personSchema,
    });

    const person = await ChatClient.create(model)
      .prompt()
      .user('Describe Ada')
      .call()
      .entity(converter);

    expect(person).toEqual({ name: 'Ada', age: 36 });
    // Format instructions appended to user message
    const userText = model.calls[0]?.getUserMessage().text ?? '';
    expect(userText).toContain('Describe Ada');
    expect(userText).toContain('JSON format');
  });

  test('entity with raw schema object', async () => {
    const model = new FakeChatModel('{"name":"Eve","age":30}');
    const person = await ChatClient.create(model)
      .prompt()
      .user('Eve')
      .call()
      .entity<{ name: string; age: number }>(personSchema);

    expect(person).toEqual({ name: 'Eve', age: 30 });
  });

  test('responseEntity returns both sides', async () => {
    const model = new FakeChatModel('{"ok":true}');
    const result = await ChatClient.create(model)
      .prompt()
      .user('q')
      .call()
      .responseEntity(mapOutputConverter());

    expect(result.entity).toEqual({ ok: true });
    expect(result.chatResponse?.content).toContain('ok');
  });

  test('useProviderStructuredOutput sets outputSchema on options', async () => {
    const model = new FakeChatModel('{"name":"Zed","age":1}');
    const converter = schemaOutputConverter({ schema: personSchema });

    await ChatClient.create(model)
      .prompt()
      .user('Zed')
      .call()
      .entity(converter, (spec) => {
        spec.useProviderStructuredOutput();
      });

    expect(model.calls[0]?.options?.outputSchema).toContain('name');
    // Native path should not append format instructions to user text
    expect(model.calls[0]?.getUserMessage().text).toBe('Zed');
  });

  test('validateSchema retries after invalid output', async () => {
    const model = new ScriptedChatModel([
      { respond: textResponse('{"name":"only"}') }, // missing age
      { respond: textResponse('{"name":"Ada","age":36}') },
    ]);

    const converter = schemaOutputConverter<{ name: string; age: number }>({
      schema: personSchema,
    });

    const person = await ChatClient.create(model)
      .prompt()
      .user('Ada')
      .call()
      .entity(converter, (spec) => {
        spec.validateSchema();
      });

    expect(person).toEqual({ name: 'Ada', age: 36 });
    expect(model.calls).toHaveLength(2);
    // Second attempt includes validation error guidance
    expect(model.calls[1]?.getUserMessage().text).toContain('Output JSON validation failed');
  });

  test('entity returns undefined for empty content', async () => {
    const model = new FakeChatModel((prompt) => {
      void prompt;
      // empty generations via empty content
      return textResponse('');
    });
    // FakeChatModel with empty string still has a generation with ""
    const entity = await ChatClient.create(model)
      .prompt()
      .user('q')
      .call()
      .entity(mapOutputConverter());
    // empty string is treated as no entity
    expect(entity).toBeUndefined();
  });
});
