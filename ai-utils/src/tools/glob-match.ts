/**
 * Minimal glob → predicate used by Glob and Grep.
 * Supports {@code *}, {@code ?}, and {@code **} (including a double-star slash prefix).
 */
export function compileGlob(pattern: string): (value: string) => boolean {
  const normalized = pattern.replace(/\\/g, '/');
  let regex = '^';
  for (let i = 0; i < normalized.length; i += 1) {
    const char = normalized.charAt(i);
    if (char === '*' && normalized[i + 1] === '*') {
      const after = normalized[i + 2];
      if (after === '/') {
        regex += '(?:.*/)?';
        i += 2;
      } else {
        regex += '.*';
        i += 1;
      }
    } else if (char === '*') {
      regex += '[^/]*';
    } else if (char === '?') {
      regex += '[^/]';
    } else if ('\\.[]{}()+-^$|'.includes(char)) {
      regex += `\\${char}`;
    } else {
      regex += char;
    }
  }
  regex += '$';
  const re = new RegExp(regex);
  return (value) => re.test(value.replace(/\\/g, '/'));
}
