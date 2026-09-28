import { type Document, document } from './document.ts';
export interface DocumentLoader<T = unknown> {
  load(
    input: T,
    options?: { id?: string; metadata?: Readonly<Record<string, unknown>> },
  ): Promise<readonly Document[]>;
}
export interface TextDocumentLoaderOptions {
  readonly encoding?: string;
}
export function textDocumentLoader(
  _options: TextDocumentLoaderOptions = {},
): DocumentLoader<string | Uint8Array> {
  return {
    async load(input, options = {}) {
      const text = typeof input === 'string' ? input : new TextDecoder().decode(input);
      return [document({ ...options, text })];
    },
  };
}
const RAW_TEXT_ELEMENTS = new Set(['script', 'style']);
function isTagNameChar(char: string): boolean {
  return (
    (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') || (char >= '0' && char <= '9')
  );
}
function readTagName(input: string, start: number): { name: string; end: number } {
  let end = start;
  while (end < input.length && isTagNameChar(input[end] as string)) end += 1;
  return { name: input.slice(start, end).toLowerCase(), end };
}
function skipAttributes(input: string, start: number): number {
  let cursor = start;
  while (cursor < input.length) {
    const char = input[cursor] as string;
    if (char === '"' || char === "'") {
      const close = input.indexOf(char, cursor + 1);
      cursor = close < 0 ? input.length : close + 1;
      continue;
    }
    if (char === '>') return cursor + 1;
    cursor += 1;
  }
  return input.length;
}
function skipRawTextElement(input: string, start: number, name: string): number {
  let cursor = start;
  while (cursor < input.length) {
    const open = input.indexOf('</', cursor);
    if (open < 0) return input.length;
    const { name: closing, end } = readTagName(input, open + 2);
    if (closing === name) return skipAttributes(input, end);
    cursor = open + 2;
  }
  return input.length;
}
function stripHtmlMarkup(input: string): string {
  let text = '';
  let cursor = 0;
  while (cursor < input.length) {
    const open = input.indexOf('<', cursor);
    if (open < 0) {
      text += input.slice(cursor);
      break;
    }
    text += input.slice(cursor, open);
    const next = input[open + 1];
    if (next === '!') {
      if (input.startsWith('<!--', open)) {
        const close = input.indexOf('-->', open + 4);
        cursor = close < 0 ? input.length : close + 3;
      } else {
        const close = input.indexOf('>', open + 2);
        cursor = close < 0 ? input.length : close + 1;
      }
      text += ' ';
      continue;
    }
    const closing = next === '/';
    const { name, end } = readTagName(input, open + (closing ? 2 : 1));
    if (name === '') {
      text += '<';
      cursor = open + 1;
      continue;
    }
    const afterTag = skipAttributes(input, end);
    const selfClosing = input[afterTag - 2] === '/';
    cursor =
      !closing && !selfClosing && RAW_TEXT_ELEMENTS.has(name)
        ? skipRawTextElement(input, afterTag, name)
        : afterTag;
    text += ' ';
  }
  return text;
}
export function htmlDocumentLoader(): DocumentLoader<string> {
  return {
    async load(input, options = {}) {
      const text = stripHtmlMarkup(input)
        .replace(/&nbsp;/gi, ' ')
        .replace(/\s+/g, ' ')
        .trim();
      return [document({ ...options, text, metadata: { ...options.metadata, format: 'html' } })];
    },
  };
}
export function pdfDocumentLoader(
  extractText: (input: Uint8Array) => string | Promise<string>,
): DocumentLoader<Uint8Array> {
  return {
    async load(input, options = {}) {
      return [
        document({
          ...options,
          text: await extractText(input),
          metadata: { ...options.metadata, format: 'pdf' },
        }),
      ];
    },
  };
}
export async function loadDocuments<T>(
  loader: DocumentLoader<T>,
  inputs: readonly T[],
  options?: { metadata?: Readonly<Record<string, unknown>> },
): Promise<Document[]> {
  const loaded = await Promise.all(inputs.map((input) => loader.load(input, options)));
  return loaded.flat();
}
