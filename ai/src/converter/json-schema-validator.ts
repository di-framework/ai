/**
 * Lightweight JSON Schema structural checks for structured-output validation.
 * Covers type / required / properties / items — enough for retry loops without
 * pulling a full draft-2020 validator dependency.
 */

/** Nesting limit for value/schema walks (untrusted model output trees). */
export const DEFAULT_JSON_SCHEMA_MAX_DEPTH = 64;

export interface SchemaValidationResult {
  readonly success: boolean;
  readonly errorMessage: string;
}

export function schemaValidationOk(): SchemaValidationResult {
  return { success: true, errorMessage: '' };
}

export function schemaValidationFailed(errorMessage: string): SchemaValidationResult {
  return { success: false, errorMessage };
}

/**
 * Validate {@code value} against a JSON Schema object or string.
 */
export function validateAgainstJsonSchema(
  value: unknown,
  schema: string | Record<string, unknown>,
  maxDepth: number = DEFAULT_JSON_SCHEMA_MAX_DEPTH,
): SchemaValidationResult {
  let schemaObj: Record<string, unknown>;
  try {
    schemaObj =
      typeof schema === 'string' ? (JSON.parse(schema) as Record<string, unknown>) : schema;
  } catch (cause) {
    return schemaValidationFailed(
      `Invalid JSON schema: ${cause instanceof Error ? cause.message : String(cause)}`,
    );
  }

  const errors: string[] = [];
  validateNode(value, schemaObj, '$', errors, 0, maxDepth);
  if (errors.length === 0) return schemaValidationOk();
  return schemaValidationFailed(errors.join('; '));
}

function validateNode(
  value: unknown,
  schema: Record<string, unknown>,
  path: string,
  errors: string[],
  depth: number,
  maxDepth: number,
): void {
  if (depth > maxDepth) {
    errors.push(`${path}: exceeded max validation depth of ${maxDepth}`);
    return;
  }

  const type = schema.type;
  if (typeof type === 'string') {
    if (!matchesType(value, type)) {
      errors.push(`${path}: expected type '${type}', got '${describeType(value)}'`);
      return;
    }
  } else if (Array.isArray(type)) {
    if (!type.some((t) => typeof t === 'string' && matchesType(value, t))) {
      errors.push(
        `${path}: expected one of types [${type.join(', ')}], got '${describeType(value)}'`,
      );
      return;
    }
  }

  if (schema.const !== undefined && !deepEqual(value, schema.const)) {
    errors.push(`${path}: expected const ${JSON.stringify(schema.const)}`);
  }

  if (Array.isArray(schema.enum) && !schema.enum.some((e) => deepEqual(value, e))) {
    errors.push(`${path}: value not in enum`);
  }

  if (type === 'object' || (value != null && typeof value === 'object' && !Array.isArray(value))) {
    const obj = value as Record<string, unknown>;
    const required = schema.required;
    if (Array.isArray(required)) {
      for (const key of required) {
        if (typeof key === 'string' && !(key in obj)) {
          errors.push(`${path}: missing required property '${key}'`);
        }
      }
    }
    const properties = schema.properties;
    if (properties && typeof properties === 'object' && !Array.isArray(properties)) {
      for (const [key, propSchema] of Object.entries(properties as Record<string, unknown>)) {
        if (key in obj && propSchema && typeof propSchema === 'object') {
          validateNode(
            obj[key],
            propSchema as Record<string, unknown>,
            `${path}.${key}`,
            errors,
            depth + 1,
            maxDepth,
          );
        }
      }
    }
  }

  if (type === 'array' || Array.isArray(value)) {
    const items = schema.items;
    if (Array.isArray(value) && items && typeof items === 'object' && !Array.isArray(items)) {
      value.forEach((item, i) => {
        validateNode(
          item,
          items as Record<string, unknown>,
          `${path}[${i}]`,
          errors,
          depth + 1,
          maxDepth,
        );
      });
    }
  }
}

function matchesType(value: unknown, type: string): boolean {
  switch (type) {
    case 'object':
      return value != null && typeof value === 'object' && !Array.isArray(value);
    case 'array':
      return Array.isArray(value);
    case 'string':
      return typeof value === 'string';
    case 'number':
      return typeof value === 'number' && Number.isFinite(value);
    case 'integer':
      return typeof value === 'number' && Number.isInteger(value);
    case 'boolean':
      return typeof value === 'boolean';
    case 'null':
      return value === null;
    default:
      return true;
  }
}

function describeType(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  try {
    return JSON.stringify(a) === JSON.stringify(b);
  } catch {
    return false;
  }
}
