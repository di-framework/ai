/**
 * Antigravity {@code plugin.json} manifest fields.
 *
 * {@link AgentPluginManifest.name} is optional; loaders default to the plugin
 * directory basename when omitted.
 */
export interface AgentPluginManifest {
  readonly name?: string;
  readonly description?: string;
  readonly version?: string;
}

export interface ParsePluginManifestOptions {
  /** Used when {@code name} is absent or blank. */
  readonly fallbackName?: string;
}

/**
 * Parse and normalize a {@code plugin.json} document.
 * Unknown fields are ignored.
 */
export function parsePluginManifest(
  raw: unknown,
  options: ParsePluginManifestOptions = {},
): AgentPluginManifest & { readonly name: string } {
  if (raw == null || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('plugin.json must be a JSON object');
  }
  const record = raw as Record<string, unknown>;
  if (Object.hasOwn(record, 'name') && typeof record.name !== 'string') {
    throw new Error('plugin.json name must be a string');
  }
  if (Object.hasOwn(record, 'description') && typeof record.description !== 'string') {
    throw new Error('plugin.json description must be a string');
  }
  if (Object.hasOwn(record, 'version') && typeof record.version !== 'string') {
    throw new Error('plugin.json version must be a string');
  }

  const nameFromManifest = typeof record.name === 'string' ? record.name.trim() : '';
  const fallback = options.fallbackName?.trim() ?? '';
  const name = nameFromManifest || fallback;
  if (!name) {
    throw new Error('plugin.json requires a non-empty name or a directory fallback name');
  }

  const description =
    typeof record.description === 'string' && record.description.trim().length > 0
      ? record.description.trim()
      : undefined;
  const version =
    typeof record.version === 'string' && record.version.trim().length > 0
      ? record.version.trim()
      : undefined;

  return { name, description, version };
}
