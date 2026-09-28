/**
 * Portable metadata filter AST.
 * Spring AI: {@code org.springframework.ai.vectorstore.filter.Filter}.
 */

export type FilterExpressionType =
  | 'AND'
  | 'OR'
  | 'EQ'
  | 'NE'
  | 'GT'
  | 'GTE'
  | 'LT'
  | 'LTE'
  | 'IN'
  | 'NIN'
  | 'NOT'
  | 'ISNULL'
  | 'ISNOTNULL';

export const FilterExpressionType = {
  AND: 'AND',
  OR: 'OR',
  EQ: 'EQ',
  NE: 'NE',
  GT: 'GT',
  GTE: 'GTE',
  LT: 'LT',
  LTE: 'LTE',
  IN: 'IN',
  NIN: 'NIN',
  NOT: 'NOT',
  ISNULL: 'ISNULL',
  ISNOTNULL: 'ISNOTNULL',
} as const satisfies Record<string, FilterExpressionType>;

/** Marker for filter AST nodes. */
export type FilterOperand = FilterKey | FilterValue | FilterExpression | FilterGroup;

export interface FilterKey {
  readonly kind: 'key';
  readonly key: string;
}

export interface FilterValue {
  readonly kind: 'value';
  readonly value: unknown;
}

export interface FilterExpression {
  readonly kind: 'expression';
  readonly type: FilterExpressionType;
  readonly left: FilterOperand;
  readonly right: FilterOperand | null;
}

export interface FilterGroup {
  readonly kind: 'group';
  readonly content: FilterExpression;
}

export function filterKey(key: string): FilterKey {
  return { kind: 'key', key };
}

export function filterValue(value: unknown): FilterValue {
  return { kind: 'value', value };
}

export function filterExpression(
  type: FilterExpressionType,
  left: FilterOperand,
  right: FilterOperand | null = null,
): FilterExpression {
  return { kind: 'expression', type, left, right };
}

export function filterGroup(content: FilterExpression): FilterGroup {
  return { kind: 'group', content };
}

export function isFilterExpression(op: FilterOperand | null | undefined): op is FilterExpression {
  return op != null && op.kind === 'expression';
}
