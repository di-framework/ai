import type { FilterExpression, FilterOperand } from './filter.ts';

/** Max nesting for filter AST evaluation (query-shaped trees). */
export const DEFAULT_FILTER_EVAL_MAX_DEPTH = 64;

/**
 * Evaluate a portable filter expression against document metadata.
 * Spring AI: {@code SimpleVectorStoreFilterExpressionEvaluator}.
 */
export function evaluateFilterExpression(
  expression: FilterExpression,
  metadata: Readonly<Record<string, unknown>>,
  maxDepth: number = DEFAULT_FILTER_EVAL_MAX_DEPTH,
): boolean {
  return evaluateExpression(expression, metadata, 0, maxDepth);
}

function evaluateOperand(
  operand: FilterOperand,
  metadata: Readonly<Record<string, unknown>>,
  depth: number,
  maxDepth: number,
): boolean {
  if (depth > maxDepth) {
    throw new Error(`Filter evaluation exceeded max depth of ${maxDepth}`);
  }
  if (operand.kind === 'group') {
    return evaluateOperand(operand.content, metadata, depth + 1, maxDepth);
  }
  if (operand.kind === 'expression') {
    return evaluateExpression(operand, metadata, depth + 1, maxDepth);
  }
  throw new Error(`Unsupported boolean operand: ${operand.kind}`);
}

function evaluateExpression(
  expression: FilterExpression,
  metadata: Readonly<Record<string, unknown>>,
  depth: number,
  maxDepth: number,
): boolean {
  if (depth > maxDepth) {
    throw new Error(`Filter evaluation exceeded max depth of ${maxDepth}`);
  }
  switch (expression.type) {
    case 'AND':
      return (
        evaluateOperand(requireLeft(expression), metadata, depth + 1, maxDepth) &&
        evaluateOperand(requireRight(expression), metadata, depth + 1, maxDepth)
      );
    case 'OR':
      return (
        evaluateOperand(requireLeft(expression), metadata, depth + 1, maxDepth) ||
        evaluateOperand(requireRight(expression), metadata, depth + 1, maxDepth)
      );
    case 'NOT':
      return !evaluateOperand(requireLeft(expression), metadata, depth + 1, maxDepth);
    case 'EQ':
      return (
        compare(
          metadataValue(requireLeft(expression), metadata),
          filterValue(requireRight(expression)),
        ) === 0
      );
    case 'NE':
      return (
        compare(
          metadataValue(requireLeft(expression), metadata),
          filterValue(requireRight(expression)),
        ) !== 0
      );
    case 'GT':
      return (
        compare(
          metadataValue(requireLeft(expression), metadata),
          filterValue(requireRight(expression)),
        ) > 0
      );
    case 'GTE':
      return (
        compare(
          metadataValue(requireLeft(expression), metadata),
          filterValue(requireRight(expression)),
        ) >= 0
      );
    case 'LT':
      return (
        compare(
          metadataValue(requireLeft(expression), metadata),
          filterValue(requireRight(expression)),
        ) < 0
      );
    case 'LTE':
      return (
        compare(
          metadataValue(requireLeft(expression), metadata),
          filterValue(requireRight(expression)),
        ) <= 0
      );
    case 'IN': {
      const metaVal = metadataValue(requireLeft(expression), metadata);
      const list = asList(filterValue(requireRight(expression)));
      return list.some((item) => compare(metaVal, item) === 0);
    }
    case 'NIN': {
      const metaVal = metadataValue(requireLeft(expression), metadata);
      const list = asList(filterValue(requireRight(expression)));
      return list.every((item) => compare(metaVal, item) !== 0);
    }
    case 'ISNULL':
      return metadataValue(requireLeft(expression), metadata) == null;
    case 'ISNOTNULL':
      return metadataValue(requireLeft(expression), metadata) != null;
    default:
      throw new Error(`Unsupported expression type: ${(expression as { type: string }).type}`);
  }
}

function requireLeft(expression: FilterExpression): FilterOperand {
  if (expression.left == null) {
    throw new Error(`Expression ${expression.type} requires a left operand`);
  }
  return expression.left;
}

function requireRight(expression: FilterExpression): FilterOperand {
  if (expression.right == null) {
    throw new Error(`Expression ${expression.type} requires a right operand`);
  }
  return expression.right;
}

function metadataValue(
  operand: FilterOperand,
  metadata: Readonly<Record<string, unknown>>,
): unknown {
  if (operand.kind !== 'key') {
    throw new Error('Expected filter key operand');
  }
  let k = operand.key;
  if (
    k.length >= 2 &&
    ((k.startsWith('"') && k.endsWith('"')) || (k.startsWith("'") && k.endsWith("'")))
  ) {
    k = k.slice(1, -1);
  }
  return Object.hasOwn(metadata, k) ? metadata[k] : null;
}

function filterValue(operand: FilterOperand): unknown {
  if (operand.kind !== 'value') {
    throw new Error('Expected filter value operand');
  }
  return operand.value;
}

function asList(value: unknown): unknown[] {
  if (Array.isArray(value)) return value;
  return [value];
}

/**
 * Compare metadata vs filter values. Numbers promote to number;
 * null is less than any non-null (SQL NULLS FIRST style for ordering).
 */
function compare(left: unknown, right: unknown): number {
  if (left == null && right == null) return 0;
  if (left == null) return -1;
  if (right == null) return 1;

  if (typeof left === 'number' && typeof right === 'number') {
    return left === right ? 0 : left < right ? -1 : 1;
  }
  if (typeof left === 'boolean' && typeof right === 'boolean') {
    return left === right ? 0 : left ? 1 : -1;
  }
  // Coerce numeric strings carefully only when both are number-like
  if (isNumeric(left) && isNumeric(right)) {
    const a = Number(left);
    const b = Number(right);
    return a === b ? 0 : a < b ? -1 : 1;
  }
  const a = String(left);
  const b = String(right);
  return a === b ? 0 : a < b ? -1 : 1;
}

function isNumeric(v: unknown): boolean {
  if (typeof v === 'number' && Number.isFinite(v)) return true;
  if (typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) {
    return true;
  }
  return false;
}
