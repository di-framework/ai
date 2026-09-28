import { basename } from 'node:path';
import type { AgentPlugin } from './load-plugins.ts';

/** Antigravity plugin name: alphanumeric, hyphens, underscores. */
const NAME_PATTERN = /^[a-zA-Z0-9-_]+$/;

/**
 * Enforce Antigravity {@code plugin.json} name rules. Returns an error message
 * or {@code undefined} when valid.
 */
export function validatePluginName(name: string | undefined): string | undefined {
  const value = name?.trim() ?? '';
  if (!value) return 'name is required';
  if (value.length > 128) return 'name must be at most 128 characters';
  if (!NAME_PATTERN.test(value)) {
    return 'name must contain only alphanumeric characters, hyphens, and underscores';
  }
  return undefined;
}

/**
 * Throw when a loaded plugin fails Antigravity name rules or directory pairing.
 */
export function validatePlugin(
  plugin: AgentPlugin,
  options: { matchDirectoryName?: boolean } = {},
): void {
  const nameError = validatePluginName(plugin.name);
  if (nameError) {
    throw new Error(`Invalid plugin (${plugin.basePath}): ${nameError}`);
  }
  if (options.matchDirectoryName === true) {
    const folder = basename(plugin.basePath);
    if (folder !== plugin.name) {
      throw new Error(
        `Invalid plugin (${plugin.basePath}): name "${plugin.name}" must match the plugin directory name "${folder}"`,
      );
    }
  }
}
