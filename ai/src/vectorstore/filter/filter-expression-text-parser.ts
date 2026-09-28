import {
  type FilterExpression,
  filterExpression,
  filterGroup,
  filterKey,
  filterValue,
} from './filter.ts';

/**
 * Minimal SQL-like filter text parser.
 * Spring AI: {@code FilterExpressionTextParser} (ANTLR); this covers common cases.
 *
 * Supported:
 * - Comparisons: `==`, `!=`, `>`, `>=`, `<`, `<=`
 * - Logic: `AND` / `&&`, `OR` / `||`, `NOT` / `!`
 * - Membership: `IN [...]`, `NIN [...]` / `NOT IN [...]`
 * - Null: `IS NULL`, `IS NOT NULL`
 * - Grouping with parentheses
 * - String (`'…'` / `"…"`), number, boolean literals
 */
/** Cap nested NOT / parentheses depth for untrusted filter text. */
export const DEFAULT_FILTER_PARSE_MAX_DEPTH = 64;

export function parseFilterExpression(
  text: string,
  maxDepth: number = DEFAULT_FILTER_PARSE_MAX_DEPTH,
): FilterExpression {
  const parser = new Parser(text, maxDepth);
  const exp = parser.parseExpression();
  parser.expectEof();
  return exp;
}

class Parser {
  private readonly tokens: Token[];
  private i = 0;
  private depth = 0;
  private readonly maxDepth: number;

  constructor(input: string, maxDepth: number) {
    this.tokens = tokenize(input);
    this.maxDepth = maxDepth;
  }

  private enter(): void {
    this.depth += 1;
    if (this.depth > this.maxDepth) {
      throw new Error(`Filter parse exceeded max depth of ${this.maxDepth}`);
    }
  }

  private leave(): void {
    this.depth -= 1;
  }

  parseExpression(): FilterExpression {
    return this.parseOr();
  }

  private parseOr(): FilterExpression {
    let left = this.parseAnd();
    while (this.matchKeyword('OR') || this.matchSymbol('||')) {
      const right = this.parseAnd();
      left = filterExpression('OR', left, right);
    }
    return left;
  }

  private parseAnd(): FilterExpression {
    let left = this.parseNot();
    while (this.matchKeyword('AND') || this.matchSymbol('&&')) {
      const right = this.parseNot();
      left = filterExpression('AND', left, right);
    }
    return left;
  }

  private parseNot(): FilterExpression {
    if (this.matchKeyword('NOT') || this.matchSymbol('!')) {
      this.enter();
      try {
        return filterExpression('NOT', this.parseNot());
      } finally {
        this.leave();
      }
    }
    return this.parsePrimary();
  }

  private parsePrimary(): FilterExpression {
    if (this.matchSymbol('(')) {
      this.enter();
      try {
        const inner = this.parseExpression();
        this.expectSymbol(')');
        return filterGroup(inner).content;
      } finally {
        this.leave();
      }
    }

    const keyTok = this.expectIdent();
    const key = filterKey(keyTok);

    if (this.matchKeyword('IS')) {
      if (this.matchKeyword('NOT')) {
        this.expectKeyword('NULL');
        return filterExpression('ISNOTNULL', key);
      }
      this.expectKeyword('NULL');
      return filterExpression('ISNULL', key);
    }

    if (this.matchKeyword('IN')) {
      return filterExpression('IN', key, filterValue(this.parseList()));
    }
    if (this.matchKeyword('NIN')) {
      return filterExpression('NIN', key, filterValue(this.parseList()));
    }
    if (this.matchKeyword('NOT')) {
      this.expectKeyword('IN');
      return filterExpression('NIN', key, filterValue(this.parseList()));
    }

    if (this.matchSymbol('==') || this.matchSymbol('=')) {
      return filterExpression('EQ', key, filterValue(this.parseLiteral()));
    }
    if (this.matchSymbol('!=')) {
      return filterExpression('NE', key, filterValue(this.parseLiteral()));
    }
    if (this.matchSymbol('>=')) {
      return filterExpression('GTE', key, filterValue(this.parseLiteral()));
    }
    if (this.matchSymbol('>')) {
      return filterExpression('GT', key, filterValue(this.parseLiteral()));
    }
    if (this.matchSymbol('<=')) {
      return filterExpression('LTE', key, filterValue(this.parseLiteral()));
    }
    if (this.matchSymbol('<')) {
      return filterExpression('LT', key, filterValue(this.parseLiteral()));
    }

    throw new Error(`Unexpected token after key "${keyTok}": ${this.peek()?.raw ?? 'EOF'}`);
  }

  private parseList(): unknown[] {
    this.expectSymbol('[');
    const values: unknown[] = [];
    if (!this.checkSymbol(']')) {
      values.push(this.parseLiteral());
      while (this.matchSymbol(',')) {
        values.push(this.parseLiteral());
      }
    }
    this.expectSymbol(']');
    return values;
  }

  private parseLiteral(): unknown {
    const t = this.advance();
    if (!t) throw new Error('Expected literal');
    if (t.type === 'string') return t.value;
    if (t.type === 'number') return t.value;
    if (t.type === 'ident') {
      const lower = String(t.value).toLowerCase();
      if (lower === 'true') return true;
      if (lower === 'false') return false;
      if (lower === 'null') return null;
      return t.value;
    }
    throw new Error(`Expected literal, got ${t.raw}`);
  }

  private peek(): Token | undefined {
    return this.tokens[this.i];
  }

  private advance(): Token | undefined {
    return this.tokens[this.i++];
  }

  private matchKeyword(kw: string): boolean {
    const t = this.peek();
    if (t?.type === 'ident' && String(t.value).toUpperCase() === kw) {
      this.i++;
      return true;
    }
    return false;
  }

  private matchSymbol(sym: string): boolean {
    const t = this.peek();
    if (t?.type === 'symbol' && t.raw === sym) {
      this.i++;
      return true;
    }
    return false;
  }

  private checkSymbol(sym: string): boolean {
    const t = this.peek();
    return t?.type === 'symbol' && t.raw === sym;
  }

  private expectKeyword(kw: string): void {
    if (!this.matchKeyword(kw)) {
      throw new Error(`Expected keyword ${kw}`);
    }
  }

  private expectSymbol(sym: string): void {
    if (!this.matchSymbol(sym)) {
      throw new Error(`Expected '${sym}'`);
    }
  }

  private expectIdent(): string {
    const t = this.advance();
    if (t?.type !== 'ident') {
      throw new Error(`Expected identifier, got ${t?.raw ?? 'EOF'}`);
    }
    return String(t.value);
  }

  expectEof(): void {
    if (this.i < this.tokens.length) {
      throw new Error(`Unexpected trailing input: ${this.peek()?.raw}`);
    }
  }
}

type Token =
  | { type: 'ident'; value: string; raw: string }
  | { type: 'string'; value: string; raw: string }
  | { type: 'number'; value: number; raw: string }
  | { type: 'symbol'; raw: string };

function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < input.length) {
    const ch = input[i]!;
    if (/\s/.test(ch)) {
      i++;
      continue;
    }

    // Multi-char symbols
    const two = input.slice(i, i + 2);
    if (
      two === '==' ||
      two === '!=' ||
      two === '>=' ||
      two === '<=' ||
      two === '&&' ||
      two === '||'
    ) {
      tokens.push({ type: 'symbol', raw: two });
      i += 2;
      continue;
    }

    if ('()[],=<>!'.includes(ch)) {
      tokens.push({ type: 'symbol', raw: ch });
      i++;
      continue;
    }

    if (ch === "'" || ch === '"') {
      const quote = ch;
      i++;
      let value = '';
      while (i < input.length && input[i] !== quote) {
        if (input[i] === '\\' && i + 1 < input.length) {
          value += input[i + 1];
          i += 2;
          continue;
        }
        value += input[i];
        i++;
      }
      if (i >= input.length) throw new Error('Unterminated string');
      i++; // closing quote
      tokens.push({ type: 'string', value, raw: `${quote}${value}${quote}` });
      continue;
    }

    if (/[0-9]/.test(ch) || (ch === '-' && /[0-9]/.test(input[i + 1] ?? ''))) {
      let raw = '';
      if (ch === '-') {
        raw += ch;
        i++;
      }
      while (i < input.length && /[0-9.]/.test(input[i]!)) {
        raw += input[i];
        i++;
      }
      tokens.push({ type: 'number', value: Number(raw), raw });
      continue;
    }

    if (/[A-Za-z_]/.test(ch)) {
      let raw = '';
      while (i < input.length && /[A-Za-z0-9_]/.test(input[i]!)) {
        raw += input[i];
        i++;
      }
      tokens.push({ type: 'ident', value: raw, raw });
      continue;
    }

    throw new Error(`Unexpected character '${ch}' at position ${i}`);
  }
  return tokens;
}
