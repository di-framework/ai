import { describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  DEFAULT_PLUGIN_DIRECTORY_CANDIDATES,
  resolvePluginPackageDirectories,
  resolvePluginSources,
  validatePluginCatalog,
} from '../src/index.ts';

function roots(): { workspace: string; userDirectory: string; explicit: string } {
  const root = mkdtempSync(join(tmpdir(), 'ai-utils-plugin-sources-'));
  const workspace = join(root, 'workspace');
  const userDirectory = join(root, 'home');
  const explicit = join(root, 'explicit');
  mkdirSync(workspace);
  mkdirSync(userDirectory);
  mkdirSync(explicit);
  return { workspace, userDirectory, explicit };
}

function writePlugin(root: string, directoryName: string, name = directoryName): string {
  const directory = join(root, directoryName);
  mkdirSync(directory, { recursive: true });
  writeFileSync(
    join(directory, 'plugin.json'),
    JSON.stringify({ name, description: `Plugin ${name}` }),
  );
  return directory;
}

describe('neutral plugin source resolution', () => {
  test('merge orders explicit, workspace, and user roots and reports shadowed plugins', () => {
    const { workspace, userDirectory, explicit } = roots();
    const workspacePlugins = join(workspace, '.agents', 'plugins');
    const userPlugins = join(userDirectory, '.agents', 'plugins');
    writePlugin(explicit, 'shared');
    writePlugin(workspacePlugins, 'shared');
    writePlugin(workspacePlugins, 'workspace-only');
    writePlugin(userPlugins, 'shared');
    writePlugin(userPlugins, 'user-only');

    const result = validatePluginCatalog({
      directories: [explicit],
      workspace,
      userDirectory,
      sourceMode: 'merge',
    });

    expect(result.plugins.map((plugin) => plugin.name)).toEqual([
      'shared',
      'workspace-only',
      'user-only',
    ]);
    expect(
      result.diagnostics
        .filter((diagnostic) => diagnostic.code === 'plugin-shadowed')
        .map((diagnostic) => diagnostic.pluginName),
    ).toEqual(['shared', 'shared']);
  });

  test('replace uses only explicit roots and empty arrays keep defaults under merge', () => {
    const { workspace, userDirectory, explicit } = roots();
    writePlugin(explicit, 'explicit-only');
    writePlugin(join(workspace, '.agents', 'plugins'), 'workspace-only');
    writePlugin(join(userDirectory, '.agents', 'plugins'), 'user-only');

    const replaced = resolvePluginSources({
      directories: [explicit],
      workspace,
      userDirectory,
      sourceMode: 'replace',
    });
    expect(replaced.sources.map((source) => source.origin)).toEqual(['explicit']);

    const mergedEmpty = resolvePluginSources({
      directories: [],
      workspace,
      userDirectory,
    });
    expect(mergedEmpty.sources.map((source) => source.origin)).toEqual(['workspace', 'user']);
  });

  test('only neutral automatic roots are candidates', () => {
    const { workspace, userDirectory } = roots();
    writePlugin(join(workspace, '.gemini', 'config', 'plugins'), 'agy-global');
    writePlugin(join(userDirectory, '.gemini', 'config', 'plugins'), 'agy-user');

    expect(DEFAULT_PLUGIN_DIRECTORY_CANDIDATES).toEqual(['.agents/plugins', '~/.agents/plugins']);
    const resolution = resolvePluginSources({ workspace, userDirectory });
    expect(resolution.sources).toEqual([]);
    expect(resolution.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
      'source-missing',
      'source-missing',
    ]);
  });

  test('package discovery prefers declarations, then .agents/plugins, then plugins, then root plugin.json', () => {
    const { workspace } = roots();
    const declaredPackage = join(workspace, 'declared-package');
    const neutralPackage = join(workspace, 'neutral-package');
    const conventionalPackage = join(workspace, 'conventional-package');
    const rootPluginPackage = join(workspace, 'root-plugin-package');
    const declared = join(declaredPackage, 'catalog');
    mkdirSync(declared, { recursive: true });
    mkdirSync(join(declaredPackage, '.agents', 'plugins'), { recursive: true });
    mkdirSync(join(neutralPackage, '.agents', 'plugins'), { recursive: true });
    mkdirSync(join(neutralPackage, 'plugins'), { recursive: true });
    mkdirSync(join(conventionalPackage, 'plugins'), { recursive: true });
    mkdirSync(rootPluginPackage, { recursive: true });
    writeFileSync(
      join(declaredPackage, 'package.json'),
      JSON.stringify({ name: 'declared', plugins: './catalog' }),
    );
    writeFileSync(join(neutralPackage, 'package.json'), JSON.stringify({ name: 'neutral' }));
    writeFileSync(
      join(conventionalPackage, 'package.json'),
      JSON.stringify({ name: 'conventional' }),
    );
    writeFileSync(join(rootPluginPackage, 'package.json'), JSON.stringify({ name: 'root-plugin' }));
    writeFileSync(
      join(rootPluginPackage, 'plugin.json'),
      JSON.stringify({ name: 'di-framework', description: 'Official-shaped root plugin' }),
    );

    expect(
      resolvePluginPackageDirectories(
        [declaredPackage, neutralPackage, conventionalPackage, rootPluginPackage],
        workspace,
      ),
    ).toEqual([
      declared,
      join(neutralPackage, '.agents', 'plugins'),
      join(conventionalPackage, 'plugins'),
      rootPluginPackage,
    ]);
    expect(
      resolvePluginSources({
        packages: [declaredPackage],
        workspace,
        sourceMode: 'replace',
      }).sources[0]?.origin,
    ).toBe('package');

    const catalog = validatePluginCatalog({
      packages: [rootPluginPackage],
      workspace,
      sourceMode: 'replace',
    });
    expect(catalog.valid).toBe(true);
    expect(catalog.plugins.map((plugin) => plugin.name)).toEqual(['di-framework']);
  });
});
