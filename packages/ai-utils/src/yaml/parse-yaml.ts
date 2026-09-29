/**
 * Small YAML subset for SKILL.md front matter: maps, lists, scalars,
 * quoted strings, {@code |} / {@code >} blocks, and nested indent.
 */

export type YamlValue = string | number | boolean | null | YamlValue[] | YamlMap;
export type YamlMap = { [key: string]: YamlValue };

export function parseYaml(text: string): YamlValue {
  const lines = toLines(text);
  if (lines.length === 0) return {};
  const first = lines[0];
  if (first == null) return {};
  if (first.text === '-' || first.text.startsWith('- ')) {
    const parsed = parseSequence(lines, 0, first.indent);
    return parsed.value;
  }
  const parsed = parseMapping(lines, 0, first.indent);
  return parsed.value;
}

export function parseYamlMap(text: string): YamlMap {
  const value = parseYaml(text);
  if (value != null && typeof value === 'object' && !Array.isArray(value)) {
    return value;
  }
  return {};
}

export function yamlValueToString(value: YamlValue | undefined): string | undefined {
  if (value == null) return undefined;
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (Array.isArray(value)) {
    return value
      .map((item) => yamlValueToString(item))
      .filter((item): item is string => item != null && item.length > 0)
      .join(', ');
  }
  return JSON.stringify(value);
}

export function flattenYamlMap(map: YamlMap): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(map)) {
    const text = yamlValueToString(value);
    if (text != null) out[key] = text;
  }
  return out;
}

interface Line {
  readonly indent: number;
  readonly text: string;
}

function toLines(text: string): Line[] {
  const out: Line[] = [];
  for (const raw of text.split(/\r?\n/)) {
    if (/^\s*#/.test(raw) || raw.trim() === '') continue;
    const indent = raw.length - raw.trimStart().length;
    out.push({ indent, text: raw.trim() });
  }
  return out;
}

function parseMapping(
  lines: Line[],
  start: number,
  indent: number,
): { value: YamlMap; next: number } {
  const map: YamlMap = {};
  let i = start;
  while (i < lines.length) {
    const line = lines[i];
    if (line == null || line.indent < indent) break;
    if (line.indent > indent) {
      throw new Error(`Unexpected indentation at "${line.text}"`);
    }
    const colon = line.text.indexOf(':');
    if (colon <= 0) {
      i += 1;
      continue;
    }
    const key = stripQuotes(line.text.slice(0, colon).trim());
    const rest = line.text.slice(colon + 1).trim();
    const nextLine = lines[i + 1];
    if (rest === '' || rest === '|' || rest === '>' || rest === '|-' || rest === '>-') {
      if (rest === '|' || rest === '|-' || rest === '>' || rest === '>-') {
        const block = parseBlockScalar(lines, i + 1, indent + 1, rest.startsWith('>'));
        map[key] = block.value;
        i = block.next;
        continue;
      }
      if (nextLine && nextLine.indent > indent) {
        if (nextLine.text.startsWith('- ')) {
          const seq = parseSequence(lines, i + 1, nextLine.indent);
          map[key] = seq.value;
          i = seq.next;
          continue;
        }
        const nested = parseMapping(lines, i + 1, nextLine.indent);
        map[key] = nested.value;
        i = nested.next;
        continue;
      }
      map[key] = '';
      i += 1;
      continue;
    }
    map[key] = parseScalar(rest);
    i += 1;
  }
  return { value: map, next: i };
}

function parseSequence(
  lines: Line[],
  start: number,
  indent: number,
): { value: YamlValue[]; next: number } {
  const items: YamlValue[] = [];
  let i = start;
  while (i < lines.length) {
    const line = lines[i];
    if (line == null || line.indent < indent) break;
    if (line.indent > indent) {
      throw new Error(`Unexpected indentation at "${line.text}"`);
    }
    if (line.text !== '-' && !line.text.startsWith('- ')) {
      break;
    }
    const rest = line.text === '-' ? '' : line.text.slice(2).trim();
    const nextLine = lines[i + 1];
    if (rest === '' || rest === '|' || rest === '>') {
      if (rest === '|' || rest === '>') {
        const block = parseBlockScalar(lines, i + 1, indent + 1, rest === '>');
        items.push(block.value);
        i = block.next;
        continue;
      }
      if (nextLine && nextLine.indent > indent) {
        if (nextLine.text.startsWith('- ')) {
          const seq = parseSequence(lines, i + 1, nextLine.indent);
          items.push(seq.value);
          i = seq.next;
          continue;
        }
        const nested = parseMapping(lines, i + 1, nextLine.indent);
        items.push(nested.value);
        i = nested.next;
        continue;
      }
      items.push(null);
      i += 1;
      continue;
    }
    if (rest.includes(': ') || rest.endsWith(':')) {
      const inlineKey = rest;
      const colon = inlineKey.indexOf(':');
      const key = stripQuotes(inlineKey.slice(0, colon).trim());
      const valuePart = inlineKey.slice(colon + 1).trim();
      const obj: YamlMap = {
        [key]: valuePart === '' ? '' : parseScalar(valuePart),
      };
      if (nextLine && nextLine.indent > indent && !nextLine.text.startsWith('- ')) {
        const nested = parseMapping(lines, i + 1, nextLine.indent);
        Object.assign(obj, nested.value);
        i = nested.next;
      } else {
        i += 1;
      }
      items.push(obj);
      continue;
    }
    items.push(parseScalar(rest));
    i += 1;
  }
  return { value: items, next: i };
}

function parseBlockScalar(
  lines: Line[],
  start: number,
  indent: number,
  folded: boolean,
): { value: string; next: number } {
  const chunks: string[] = [];
  let i = start;
  while (i < lines.length) {
    const line = lines[i];
    if (line == null || line.indent < indent) break;
    chunks.push(line.text);
    i += 1;
  }
  return { value: folded ? chunks.join(' ') : chunks.join('\n'), next: i };
}

function parseScalar(raw: string): YamlValue {
  const trimmed = raw.trim();
  if (trimmed === '~' || trimmed === 'null' || trimmed === 'Null') return null;
  if (trimmed === 'true' || trimmed === 'True') return true;
  if (trimmed === 'false' || trimmed === 'False') return false;
  if (
    (trimmed.startsWith('[') && trimmed.endsWith(']')) ||
    (trimmed.startsWith('{') && trimmed.endsWith('}'))
  ) {
    return parseFlow(trimmed);
  }
  if (/^[+-]?\d+$/.test(trimmed)) return Number(trimmed);
  if (/^[+-]?\d+\.\d+$/.test(trimmed)) return Number(trimmed);
  return stripQuotes(trimmed);
}

function parseFlow(text: string): YamlValue {
  try {
    const jsonish = text
      .replace(/([{,]\s*)([A-Za-z_][\w-]*)\s*:/g, '$1"$2":')
      .replace(/'([^']*)'/g, '"$1"');
    return JSON.parse(jsonish) as YamlValue;
  } catch {
    return stripQuotes(text);
  }
}

function stripQuotes(value: string): string {
  if (value.length >= 2) {
    const start = value[0];
    const end = value[value.length - 1];
    if ((start === '"' && end === '"') || (start === "'" && end === "'")) {
      return value.slice(1, -1);
    }
  }
  return value;
}
