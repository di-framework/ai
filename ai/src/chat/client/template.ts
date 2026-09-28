/**
 * Minimal template renderer for {@code {var}} placeholders.
 * Spring AI uses StringTemplate by default; this is a light TS stand-in.
 */
export function renderTemplate(
  template: string,
  params: Readonly<Record<string, unknown>>,
): string {
  return template.replace(/\{([^{}]+)\}/g, (match, key: string) => {
    const trimmed = key.trim();
    if (Object.hasOwn(params, trimmed)) {
      const value = params[trimmed];
      return value == null ? '' : String(value);
    }
    return match;
  });
}
