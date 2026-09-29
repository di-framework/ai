import {
  type FilterExpression,
  type FilterOperand,
  filterExpression,
  filterGroup,
  filterKey,
  filterValue,
} from './filter.ts';

/**
 * Fluent DSL for portable filter expressions.
 * Spring AI: {@code FilterExpressionBuilder}.
 *
 * @example
 * ```ts
 * const b = new FilterExpressionBuilder();
 * const exp = b.and(b.eq("country", "UK"), b.gte("year", 2020)).build();
 * ```
 */
export class FilterExpressionBuilder {
  eq(key: string, value: unknown): FilterOp {
    return new FilterOp(filterExpression('EQ', filterKey(key), filterValue(value)));
  }

  ne(key: string, value: unknown): FilterOp {
    return new FilterOp(filterExpression('NE', filterKey(key), filterValue(value)));
  }

  gt(key: string, value: unknown): FilterOp {
    return new FilterOp(filterExpression('GT', filterKey(key), filterValue(value)));
  }

  gte(key: string, value: unknown): FilterOp {
    return new FilterOp(filterExpression('GTE', filterKey(key), filterValue(value)));
  }

  lt(key: string, value: unknown): FilterOp {
    return new FilterOp(filterExpression('LT', filterKey(key), filterValue(value)));
  }

  lte(key: string, value: unknown): FilterOp {
    return new FilterOp(filterExpression('LTE', filterKey(key), filterValue(value)));
  }

  and(left: FilterOp, right: FilterOp): FilterOp {
    return new FilterOp(filterExpression('AND', left.operand, right.operand));
  }

  or(left: FilterOp, right: FilterOp): FilterOp {
    return new FilterOp(filterExpression('OR', left.operand, right.operand));
  }

  in(key: string, ...values: unknown[]): FilterOp {
    return new FilterOp(filterExpression('IN', filterKey(key), filterValue(values)));
  }

  nin(key: string, ...values: unknown[]): FilterOp {
    return new FilterOp(filterExpression('NIN', filterKey(key), filterValue(values)));
  }

  isNull(key: string): FilterOp {
    return new FilterOp(filterExpression('ISNULL', filterKey(key)));
  }

  isNotNull(key: string): FilterOp {
    return new FilterOp(filterExpression('ISNOTNULL', filterKey(key)));
  }

  group(content: FilterOp): FilterOp {
    return new FilterOp(filterGroup(content.build()));
  }

  not(content: FilterOp): FilterOp {
    return new FilterOp(filterExpression('NOT', content.operand));
  }
}

export class FilterOp {
  constructor(readonly operand: FilterOperand) {}

  build(): FilterExpression {
    if (this.operand.kind === 'group') {
      return this.operand.content;
    }
    if (this.operand.kind === 'expression') {
      return this.operand;
    }
    throw new Error(`Invalid expression: ${this.operand.kind}`);
  }
}
