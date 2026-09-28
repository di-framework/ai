import { describe, expect, test } from 'bun:test';
import { chmodSync, mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  existingPluginDirectories,
  loadPluginDirectory,
  loadPluginsDirectories,
  loadPluginsDirectory,
  parseMcpConfig,
  parsePluginManifest,
  resolvePluginPackageDirectories,
  validatePlugin,
  validatePluginCatalog,
  validatePluginDirectory,
  validatePluginName,
  validateResolvedPluginCatalog,
} from '../src/index.ts';

function root(): string {
  return mkdtempSync(join(tmpdir(), 'plugin-coverage-'));
}

function writeMinimalPlugin(pluginsRoot: string, name: string): string {
  const directory = join(pluginsRoot, name);
  mkdirSync(directory, { recursive: true });
  writeFileSync(join(directory, 'plugin.json'), JSON.stringify({ name }));
  return directory;
}

describe('plugin coverage edges', () => {
  test('existingPluginDirectories skips missing and non-directory candidates', () => {
    const workspace = root();
    const plugins = join(workspace, '.agents', 'plugins');
    mkdirSync(plugins, { recursive: true });
    const file = join(workspace, 'not-a-dir');
    writeFileSync(file, 'file');
    expect(existingPluginDirectories([plugins, join(workspace, 'missing'), file])).toEqual([
      plugins,
    ]);
  });

  test('loadPluginDirectory and loadPluginsDirectory error paths', () => {
    const workspace = root();
    const missing = join(workspace, 'missing');
    const file = join(workspace, 'file');
    writeFileSync(file, 'not a directory');
    expect(() => loadPluginDirectory(missing)).toThrow(/does not exist/);
    expect(() => loadPluginDirectory(file)).toThrow(/not a directory/);
    expect(() => loadPluginsDirectory(missing)).toThrow(/does not exist/);
    expect(() => loadPluginsDirectory(file)).toThrow(/not a directory/);

    const empty = join(workspace, 'empty');
    mkdirSync(empty);
    expect(() => loadPluginDirectory(empty)).toThrow(/missing plugin\.json/);

    const badJson = join(workspace, 'bad-json');
    mkdirSync(badJson);
    writeFileSync(join(badJson, 'plugin.json'), '{');
    expect(() => loadPluginDirectory(badJson)).toThrow(/Unable to read plugin\.json/);

    const pluginsRoot = join(workspace, 'plugins');
    mkdirSync(pluginsRoot);
    writeFileSync(join(pluginsRoot, 'readme.txt'), 'skip');
    const brokenChild = join(pluginsRoot, 'broken');
    symlinkSync(join(workspace, 'missing-target'), brokenChild);
    writeMinimalPlugin(pluginsRoot, 'ok');
    mkdirSync(join(pluginsRoot, 'no-manifest'), { recursive: true });
    expect(loadPluginsDirectory(pluginsRoot).map((plugin) => plugin.name)).toEqual(['ok']);
    expect(loadPluginsDirectories([pluginsRoot]).map((plugin) => plugin.name)).toEqual(['ok']);

    const singleRoot = writeMinimalPlugin(workspace, 'single-root');
    expect(loadPluginsDirectory(singleRoot).map((plugin) => plugin.name)).toEqual(['single-root']);
  });

  test('loadPluginDirectory rejects invalid mcp_config and hooks', () => {
    const workspace = root();
    const badMcpJson = writeMinimalPlugin(workspace, 'bad-mcp-json');
    writeFileSync(join(badMcpJson, 'mcp_config.json'), '{');
    expect(() => loadPluginDirectory(badMcpJson)).toThrow(/Unable to parse mcp_config/);

    const badMcpShape = writeMinimalPlugin(workspace, 'bad-mcp-shape');
    writeFileSync(join(badMcpShape, 'mcp_config.json'), JSON.stringify({ mcpServers: [] }));
    expect(() => loadPluginDirectory(badMcpShape)).toThrow(/Invalid mcp_config/);

    const badHooksJson = writeMinimalPlugin(workspace, 'bad-hooks-json');
    writeFileSync(join(badHooksJson, 'hooks.json'), '{');
    expect(() => loadPluginDirectory(badHooksJson)).toThrow(/Unable to parse hooks/);

    const badHooksShape = writeMinimalPlugin(workspace, 'bad-hooks-shape');
    writeFileSync(join(badHooksShape, 'hooks.json'), JSON.stringify([]));
    expect(() => loadPluginDirectory(badHooksShape)).toThrow(/hooks\.json must be a JSON object/);

    const withUnreadableRule = writeMinimalPlugin(workspace, 'rules-skip');
    mkdirSync(join(withUnreadableRule, 'rules'));
    const rule = join(withUnreadableRule, 'rules', 'secret.md');
    writeFileSync(rule, 'secret');
    chmodSync(rule, 0o000);
    expect(loadPluginDirectory(withUnreadableRule).rules).toEqual([]);
    chmodSync(rule, 0o600);
  });

  test('parsePluginManifest and validatePlugin cover field and name rules', () => {
    expect(() => parsePluginManifest(null)).toThrow(/JSON object/);
    expect(() => parsePluginManifest({ description: 1 }, { fallbackName: 'x' })).toThrow(
      /description must be a string/,
    );
    expect(() => parsePluginManifest({ version: 1 }, { fallbackName: 'x' })).toThrow(
      /version must be a string/,
    );
    expect(() => parsePluginManifest({})).toThrow(/non-empty name/);
    expect(
      parsePluginManifest({ name: '  ', description: '  ', version: '  ' }, { fallbackName: 'ok' }),
    ).toEqual({
      name: 'ok',
      description: undefined,
      version: undefined,
    });

    expect(validatePluginName(undefined)).toMatch(/required/);
    expect(validatePluginName('a'.repeat(129))).toMatch(/at most 128/);
    expect(validatePluginName('bad name')).toMatch(/alphanumeric/);
    expect(validatePluginName('good_Name-1')).toBeUndefined();

    expect(() =>
      validatePlugin({
        name: '',
        basePath: '/plugins/x',
        manifestPath: '/plugins/x/plugin.json',
        rules: [],
      }),
    ).toThrow(/Invalid plugin/);
    expect(() =>
      validatePlugin(
        {
          name: 'other',
          basePath: '/plugins/di-framework',
          manifestPath: '/plugins/di-framework/plugin.json',
          rules: [],
        },
        { matchDirectoryName: true },
      ),
    ).toThrow(/must match the plugin directory name/);
    expect(() =>
      validatePlugin({
        name: 'di-framework',
        basePath: '/plugins/di-framework',
        manifestPath: '/plugins/di-framework/plugin.json',
        rules: [],
      }),
    ).not.toThrow();
    expect(() =>
      validatePlugin(
        {
          name: 'di-framework',
          basePath: '/plugins/di-framework',
          manifestPath: '/plugins/di-framework/plugin.json',
          rules: [],
        },
        { matchDirectoryName: true },
      ),
    ).not.toThrow();
  });

  test('parseMcpConfig covers root and server field validations', () => {
    expect(parseMcpConfig(null).errors[0]?.code).toBe('mcp-config-invalid-root');
    expect(parseMcpConfig({}).errors[0]?.code).toBe('mcp-config-missing-servers');
    expect(parseMcpConfig({ mcpServers: null }).errors[0]?.code).toBe('mcp-config-servers-invalid');
    expect(parseMcpConfig({ mcpServers: { a: null } }).errors[0]?.code).toBe(
      'mcp-config-server-invalid',
    );
    expect(parseMcpConfig({ mcpServers: { a: {} } }).errors[0]?.code).toBe(
      'mcp-config-transport-missing',
    );
    expect(parseMcpConfig({ mcpServers: { a: { command: 1 } } }).errors[0]?.message).toMatch(
      /command must be a string/,
    );
    expect(parseMcpConfig({ mcpServers: { a: { serverUrl: 1 } } }).errors[0]?.message).toMatch(
      /serverUrl must be a string/,
    );
    expect(
      parseMcpConfig({ mcpServers: { a: { command: 'node', args: [1] } } }).errors[0]?.message,
    ).toMatch(/args must be a string array/);
    expect(
      parseMcpConfig({ mcpServers: { a: { command: 'node', env: { X: 1 } } } }).errors[0]?.message,
    ).toMatch(/env must be a string map/);
    expect(
      parseMcpConfig({ mcpServers: { a: { command: 'node', headers: { X: 1 } } } }).errors[0]
        ?.message,
    ).toMatch(/headers must be a string map/);
    expect(
      parseMcpConfig({ mcpServers: { a: { command: 'node', cwd: 1 } } }).errors[0]?.message,
    ).toMatch(/cwd must be a string/);
    expect(
      parseMcpConfig({ mcpServers: { a: { command: 'node', disabled: 'yes' } } }).errors[0]
        ?.message,
    ).toMatch(/disabled must be a boolean/);
    expect(
      parseMcpConfig({ mcpServers: { a: { command: 'node', disabledTools: [1] } } }).errors[0]
        ?.message,
    ).toMatch(/disabledTools must be a string array/);
    expect(
      parseMcpConfig({ mcpServers: { a: { command: 'node', description: 1 } } }).errors[0]?.message,
    ).toMatch(/description must be a string/);
    expect(
      parseMcpConfig({ mcpServers: { a: { command: 'node', oauth: [] } } }).errors[0]?.message,
    ).toMatch(/oauth must be an object/);
    expect(
      parseMcpConfig({ mcpServers: { a: { command: 'node', authProviderType: 1 } } }).errors[0]
        ?.message,
    ).toMatch(/authProviderType must be a string/);

    const ok = parseMcpConfig({
      mcpServers: {
        remote: {
          serverUrl: 'https://example.com/mcp',
          headers: { Authorization: 'Bearer x' },
          disabled: false,
          disabledTools: ['secret'],
          description: 'docs',
          oauth: { clientId: 'id' },
          authProviderType: 'google_credentials',
          cwd: '/tmp',
          env: { A: '1' },
          args: ['--stdio'],
          command: 'node',
        },
      },
    });
    expect(ok.errors).toEqual([]);
    expect(ok.config?.mcpServers.remote?.serverUrl).toBe('https://example.com/mcp');
  });

  test('validatePluginDirectory covers symlink and shape failures', () => {
    const workspace = root();
    const missing = join(workspace, 'missing');
    expect(validatePluginDirectory(missing).diagnostics[0]?.code).toBe('plugin-manifest-missing');

    const file = join(workspace, 'file');
    writeFileSync(file, 'not a directory');
    expect(validatePluginDirectory(file).diagnostics[0]?.code).toBe('plugin-manifest-missing');

    const brokenDir = join(workspace, 'broken-dir');
    symlinkSync(join(workspace, 'missing-target'), brokenDir);
    expect(validatePluginDirectory(brokenDir).diagnostics[0]?.code).toBe('plugin-child-unreadable');

    const brokenManifest = join(workspace, 'broken-manifest');
    mkdirSync(brokenManifest);
    symlinkSync(join(workspace, 'missing-plugin.json'), join(brokenManifest, 'plugin.json'));
    expect(validatePluginDirectory(brokenManifest).diagnostics[0]?.code).toBe(
      'plugin-child-unreadable',
    );

    const invalidManifest = join(workspace, 'invalid-manifest');
    mkdirSync(invalidManifest);
    writeFileSync(join(invalidManifest, 'plugin.json'), JSON.stringify({ name: 1 }));
    expect(validatePluginDirectory(invalidManifest).diagnostics[0]?.code).toBe(
      'plugin-manifest-invalid',
    );

    const badMcpJson = writeMinimalPlugin(workspace, 'mcp-json');
    writeFileSync(join(badMcpJson, 'mcp_config.json'), '{');
    expect(
      validatePluginDirectory(badMcpJson).diagnostics.some(
        (d) => d.code === 'plugin-mcp-config-invalid',
      ),
    ).toBe(true);

    const badHooksJson = writeMinimalPlugin(workspace, 'hooks-json');
    writeFileSync(join(badHooksJson, 'hooks.json'), '{');
    expect(
      validatePluginDirectory(badHooksJson).diagnostics.some(
        (d) => d.code === 'plugin-hooks-invalid',
      ),
    ).toBe(true);

    const unreadableRule = writeMinimalPlugin(workspace, 'rule-unreadable');
    mkdirSync(join(unreadableRule, 'rules'));
    const rule = join(unreadableRule, 'rules', 'locked.md');
    writeFileSync(rule, 'locked');
    chmodSync(rule, 0o000);
    const ruleResult = validatePluginDirectory(unreadableRule);
    chmodSync(rule, 0o600);
    expect(ruleResult.diagnostics.some((d) => d.code === 'plugin-rules-unreadable')).toBe(true);
  });

  test('resolvePluginPackageDirectories throws for unresolved package names', () => {
    const workspace = root();
    writeFileSync(join(workspace, 'package.json'), JSON.stringify({ name: 'workspace' }));
    expect(() =>
      resolvePluginPackageDirectories(['@di-framework/does-not-exist-plugin-pkg'], workspace),
    ).toThrow(/Cannot resolve plugin package/);

    const brokenPkg = join(workspace, 'broken-pkg');
    mkdirSync(brokenPkg);
    writeFileSync(join(brokenPkg, 'package.json'), '{');
    expect(resolvePluginPackageDirectories([brokenPkg], workspace)).toEqual([]);
  });

  test('validateResolvedPluginCatalog maps source diagnostics and broken children', () => {
    const workspace = root();
    const pluginsRoot = join(workspace, 'plugins');
    mkdirSync(pluginsRoot);
    const valid = writeMinimalPlugin(pluginsRoot, 'valid');
    writeFileSync(
      join(valid, 'mcp_config.json'),
      JSON.stringify({ mcpServers: { docs: { command: 'node' } } }),
    );
    writeFileSync(join(valid, 'hooks.json'), JSON.stringify({ PreToolUse: [] }));
    const brokenChild = join(pluginsRoot, 'broken-child');
    symlinkSync(join(workspace, 'missing-target'), brokenChild);
    const badChild = writeMinimalPlugin(pluginsRoot, 'bad-child');
    writeFileSync(join(badChild, 'plugin.json'), JSON.stringify({ name: 1 }));

    const skillsRoot = join(valid, 'skills');
    const brokenSkillLink = join(skillsRoot, 'broken-skill');
    mkdirSync(skillsRoot);
    symlinkSync(join(workspace, 'missing-skill'), brokenSkillLink);

    const result = validatePluginCatalog({
      directories: [pluginsRoot],
      sourceMode: 'replace',
      workspace,
    });
    expect(result.plugins.some((plugin) => plugin.name === 'valid')).toBe(true);
    expect(result.diagnostics.some((d) => d.code === 'plugin-manifest-invalid')).toBe(true);

    const withSourceDiagnostics = validateResolvedPluginCatalog({
      directories: [pluginsRoot],
      sources: [
        {
          path: pluginsRoot,
          realPath: pluginsRoot,
          origin: 'explicit',
          precedence: 0,
          kind: 'directory',
        },
      ],
      diagnostics: [
        {
          code: 'source-duplicate',
          severity: 'warning',
          path: pluginsRoot,
          origin: 'explicit',
          precedence: 1,
          message: 'duplicate source',
          duplicateOf: pluginsRoot,
        },
      ],
    });
    expect(withSourceDiagnostics.diagnostics.some((d) => d.code === 'source-duplicate')).toBe(true);
  });
});
