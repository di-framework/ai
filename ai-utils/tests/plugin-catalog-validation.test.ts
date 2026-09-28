import { describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  loadPluginDirectory,
  loadPluginsDirectory,
  parsePluginManifest,
  resolvePluginSources,
  validatePluginCatalog,
  validatePluginDefinition,
  validatePluginDirectory,
  validatePluginsDirectory,
  validateResolvedPluginCatalog,
} from '../src/index.ts';

function root(): string {
  return mkdtempSync(join(tmpdir(), 'plugin-catalog-validation-'));
}

function writePlugin(
  pluginsRoot: string,
  directoryName: string,
  options: {
    name?: string;
    description?: string;
    version?: string;
    mcpConfig?: unknown;
    hooks?: unknown;
    rules?: Record<string, string>;
    skills?: Record<string, string>;
  } = {},
): string {
  const directory = join(pluginsRoot, directoryName);
  mkdirSync(directory, { recursive: true });
  writeFileSync(
    join(directory, 'plugin.json'),
    JSON.stringify({
      name: options.name ?? directoryName,
      description: options.description ?? `Plugin ${directoryName}`,
      version: options.version,
    }),
  );
  if (options.mcpConfig !== undefined) {
    writeFileSync(join(directory, 'mcp_config.json'), JSON.stringify(options.mcpConfig));
  }
  if (options.hooks !== undefined) {
    writeFileSync(join(directory, 'hooks.json'), JSON.stringify(options.hooks));
  }
  if (options.rules) {
    mkdirSync(join(directory, 'rules'), { recursive: true });
    for (const [name, content] of Object.entries(options.rules)) {
      writeFileSync(join(directory, 'rules', `${name}.md`), content);
    }
  }
  if (options.skills) {
    for (const [skillName, body] of Object.entries(options.skills)) {
      const skillDir = join(directory, 'skills', skillName);
      mkdirSync(skillDir, { recursive: true });
      writeFileSync(
        join(skillDir, 'SKILL.md'),
        `---\nname: ${skillName}\ndescription: Use ${skillName}.\n---\n${body}`,
      );
    }
  }
  return directory;
}

function codes(result: { diagnostics: readonly { code: string }[] }): string[] {
  return result.diagnostics.map((diagnostic) => diagnostic.code);
}

describe('plugin catalog validation', () => {
  test('parses manifests and defaults name to the directory', () => {
    expect(parsePluginManifest({}, { fallbackName: 'di-framework' }).name).toBe('di-framework');
    expect(() => parsePluginManifest([])).toThrow(/JSON object/);
    expect(() => parsePluginManifest({ name: 1 })).toThrow(/name must be a string/);
  });

  test('loads a complete plugin bundle', () => {
    const workspace = root();
    const plugin = writePlugin(workspace, 'di-framework', {
      version: '1.0.0',
      mcpConfig: {
        mcpServers: {
          docs: { command: 'node', args: ['./dist/index.js'] },
        },
      },
      hooks: { PreToolUse: [] },
      rules: { AGENTS: '# conventions' },
      skills: { 'api-guide': 'Use the API carefully.' },
    });

    const loaded = loadPluginDirectory(plugin);
    expect(loaded.name).toBe('di-framework');
    expect(loaded.version).toBe('1.0.0');
    expect(loaded.mcpConfig?.mcpServers.docs?.command).toBe('node');
    expect(loaded.hooks).toEqual({ PreToolUse: [] });
    expect(loaded.rules.map((rule) => rule.name)).toEqual(['AGENTS']);
    expect(loaded.skillsDirectory).toBe(join(plugin, 'skills'));
    expect(loadPluginsDirectory(workspace).map((item) => item.name)).toEqual(['di-framework']);
  });

  test('reports missing and invalid manifests', () => {
    const workspace = root();
    const empty = join(workspace, 'empty');
    const badJson = join(workspace, 'bad-json');
    mkdirSync(empty);
    mkdirSync(badJson);
    writeFileSync(join(badJson, 'plugin.json'), '{');

    expect(codes(validatePluginDirectory(empty))).toEqual(['plugin-manifest-missing']);
    expect(codes(validatePluginDirectory(badJson))).toEqual(['plugin-manifest-invalid']);
  });

  test('reports invalid plugin names and mcp_config / hooks shapes', () => {
    const workspace = root();
    const badName = writePlugin(workspace, 'Bad Name!', { name: 'Bad Name!' });
    const badMcp = writePlugin(workspace, 'bad-mcp', {
      mcpConfig: { mcpServers: { docs: { args: ['x'] } } },
    });
    const badHooks = writePlugin(workspace, 'bad-hooks', { hooks: ['not-an-object'] });

    expect(codes(validatePluginDirectory(badName))).toContain('plugin-name-invalid');
    expect(codes(validatePluginDirectory(badMcp))).toContain('plugin-mcp-config-invalid');
    expect(codes(validatePluginDirectory(badHooks))).toContain('plugin-hooks-invalid');
  });

  test('validates nested skills and keeps valid plugins', () => {
    const workspace = root();
    const plugin = writePlugin(workspace, 'with-skills', {
      skills: { 'good-skill': 'ok' },
    });
    mkdirSync(join(plugin, 'skills', 'broken'), { recursive: true });

    const result = validatePluginDirectory(plugin);
    expect(result.valid).toBe(false);
    expect(result.plugins[0]?.name).toBe('with-skills');
    expect(codes(result)).toContain('skill-entrypoint-missing');
  });

  test('distinguishes same-source duplicates from lower-precedence shadows', () => {
    const workspace = root();
    const explicit = join(workspace, 'explicit');
    const lower = join(workspace, 'lower');
    writePlugin(explicit, 'first-directory', { name: 'shared' });
    writePlugin(explicit, 'second-directory', { name: 'shared' });
    writePlugin(lower, 'shared');
    const resolution = resolvePluginSources({
      workspace,
      directories: [explicit, lower],
      sourceMode: 'replace',
    });

    const result = validateResolvedPluginCatalog(resolution);

    expect(result.plugins.map((plugin) => plugin.name)).toEqual(['shared']);
    expect(codes(result)).toContain('plugin-duplicate');
    expect(codes(result)).toContain('plugin-shadowed');
    expect(validatePluginsDirectory(lower).plugins.map((plugin) => plugin.name)).toEqual([
      'shared',
    ]);
  });

  test('reports a source that becomes unreadable after resolution', () => {
    const workspace = root();
    const missing = join(workspace, 'removed-after-resolution');
    const result = validateResolvedPluginCatalog({
      directories: [missing],
      diagnostics: [],
      sources: [
        {
          path: missing,
          realPath: missing,
          origin: 'explicit',
          precedence: 0,
          kind: 'directory',
        },
      ],
    });

    expect(codes(result)).toEqual(['source-unreadable']);
  });

  test('validatePluginDefinition can require directory name pairing', () => {
    const result = validatePluginDefinition(
      {
        name: 'other',
        basePath: '/plugins/di-framework',
        manifestPath: '/plugins/di-framework/plugin.json',
        rules: [],
      },
      { matchDirectoryName: true },
    );
    expect(codes(result)).toEqual(['plugin-name-directory-mismatch']);
  });

  test('validatePluginCatalog resolves defaults through workspace roots', () => {
    const workspace = root();
    const userDirectory = join(workspace, 'home');
    mkdirSync(join(userDirectory, '.agents', 'plugins'), { recursive: true });
    writePlugin(join(workspace, '.agents', 'plugins'), 'workspace-plugin');
    const result = validatePluginCatalog({
      workspace,
      userDirectory,
    });
    expect(result.plugins.map((plugin) => plugin.name)).toEqual(['workspace-plugin']);
    expect(result.valid).toBe(true);
    expect(codes(result)).toEqual([]);
  });
});
