import { afterEach, describe, expect, spyOn, test } from 'bun:test';
import * as fs from 'node:fs';
import { chmodSync, mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { join } from 'node:path';
import { ChatClient, functionToolCallback, ScriptedChatModel } from '@di-framework/ai';
import {
  agentSkill,
  assertPathAllowed,
  bashTool,
  compileGlob,
  createSkillsAgent,
  createSkillsRuntime,
  editTool,
  existingSkillDirectories,
  formatMemorySystemPrompt,
  globTool,
  grepTool,
  listDirectoryTool,
  loadSkillsDirectory,
  memoryTools,
  parseSkillMarkdown,
  parseYaml,
  parseYamlMap,
  readTool,
  resolveSkillPackageDirectories,
  SkillsAgent,
  SkillsTool,
  SkillsToolbox,
  validateSkill,
  webFetchTool,
  webSearchTool,
  writeTool,
} from '../src/index.ts';

afterEach(() => {
  // bun spyOn restores per-test if we call mockRestore on each spy we create
});

describe('yaml remaining branches', () => {
  test('parses empty, sequences, flow, scalars, and bad indent', () => {
    expect(parseYaml('')).toEqual({});
    expect(parseYaml('# only comment')).toEqual({});
    expect(parseYaml('- a\n- b')).toEqual(['a', 'b']);
    expect(parseYamlMap('- a')).toEqual({});
    expect(parseYaml('empty:\n')).toEqual({ empty: '' });
    expect(parseYaml('folded: >\n  one\n  two')).toEqual({ folded: 'one two' });
    expect(parseYaml('strip: |-\n  keep')).toEqual({ strip: 'keep' });
    expect(parseYaml('wide: >-\n  x')).toEqual({ wide: 'x' });
    expect(parseYaml('n: 1.5\nok: True\noff: False\nz: ~')).toEqual({
      n: 1.5,
      ok: true,
      off: false,
      z: null,
    });
    expect(parseYaml('obj: {a: 1}')).toEqual({ obj: { a: 1 } });
    expect(parseYaml('arr: [1, 2]')).toEqual({ arr: [1, 2] });
    expect(parseYaml('bad: {not json')).toEqual({ bad: '{not json' });
    expect(parseYaml('- name: ada\n  age: 2')).toEqual([{ name: 'ada', age: 2 }]);
    expect(parseYaml('- |\n  block')).toEqual(['block']);
    expect(parseYaml('- >\n  fold')).toEqual(['fold']);
    expect(parseYaml('- name: ada')).toEqual([{ name: 'ada' }]);
    expect(parseYaml('list:\n  - |\n    block')).toEqual({ list: ['block'] });
    expect(parseYaml('list:\n  - >\n    fold')).toEqual({ list: ['fold'] });
    expect(parseYaml('-\n  name: ada')).toEqual([{ name: 'ada' }]);
    expect(parseYaml('-\n  - nested')).toEqual([['nested']]);
    expect(parseYaml('-')).toEqual([null]);
    expect(parseYaml('flow: {not json}')).toEqual({ flow: '{not json}' });
    expect(() => parseYaml('name: x\n    oops: y')).toThrow(/Unexpected indentation/);
    expect(() => parseYaml('- a\n    - b')).toThrow(/Unexpected indentation/);
  });
});

describe('skill parse / load / resolve remaining branches', () => {
  test('agentSkill serializes allowed-tools and parseYamlMap errors become empty yaml', () => {
    const skill = agentSkill({
      name: 'xlsx',
      description: 'Build a spreadsheet when asked.',
      content: 'xlsx',
      allowedTools: ['Read', 'Grep'],
    });
    expect(skill.frontMatter['allowed-tools']).toBe('Read, Grep');
    const parsed = parseSkillMarkdown(
      `---
name: x
    bad: y
---
body
`,
      { fallbackName: 'x' },
    );
    expect(parsed.name).toBe('x');
  });

  test('existingSkillDirectories skips missing and keeps real dirs', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-exist-'));
    mkdirSync(join(root, 'keep'));
    expect(
      existingSkillDirectories([join(root, 'missing'), join(root, 'keep'), join(root, 'file')]),
    ).toEqual([join(root, 'keep')]);
    writeFileSync(join(root, 'file'), 'x');
    expect(existingSkillDirectories([join(root, 'file')])).toEqual([]);
  });

  test('loadSkillsDirectory skips unreadable children', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-walk-'));
    const hidden = join(root, 'hidden');
    mkdirSync(hidden);
    writeFileSync(join(hidden, 'SKILL.md'), '---\nname: hidden\ndescription: d\n---\n');
    chmodSync(hidden, 0);
    try {
      expect(() => loadSkillsDirectory(root)).not.toThrow();
    } finally {
      chmodSync(hidden, 0o755);
    }
  });

  test('resolveSkillPackageDirectories covers fallbacks and errors', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-pkg2-'));
    const declared = join(root, 'declared');
    mkdirSync(declared);
    writeFileSync(join(declared, 'package.json'), '{not json');
    expect(resolveSkillPackageDirectories([declared])).toEqual([]);

    const withMissing = join(root, 'miss');
    mkdirSync(withMissing);
    writeFileSync(
      join(withMissing, 'package.json'),
      JSON.stringify({ name: 'm', skills: ['./noop', 1] }),
    );
    expect(resolveSkillPackageDirectories([withMissing])).toEqual([]);

    const fallback = join(root, 'fb');
    mkdirSync(join(fallback, '.agents', 'skills'), { recursive: true });
    writeFileSync(join(fallback, 'package.json'), JSON.stringify({ name: 'fb' }));
    expect(resolveSkillPackageDirectories([fallback]).some((d) => d.endsWith('skills'))).toBe(true);

    expect(resolveSkillPackageDirectories(['./fb'], root).length).toBeGreaterThan(0);
    expect(() => resolveSkillPackageDirectories(['not-a-real-pkg-zzz'], root)).toThrow(
      /Cannot resolve skill package/,
    );
  });
});

describe('builders remaining methods', () => {
  test('SkillsToolbox.builder fluent surface and extras', () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-fluent-'));
    mkdirSync(join(root, 'code-reviewer'));
    writeFileSync(
      join(root, 'code-reviewer', 'SKILL.md'),
      `---
name: code-reviewer
description: Reviews TypeScript when asked to review or audit code.
---
# r
`,
    );
    const extra = mkdtempSync(join(tmpdir(), 'ai-utils-extra-'));
    let activated = '';
    const tools = SkillsToolbox.builder()
      .addSkill(
        agentSkill({
          name: 'inline',
          description: 'Inline helper when asked for inline help.',
          content: 'x',
        }),
      )
      .addSkills([])
      .addSkillsDirectories([root])
      .addSkillsFile(join(root, 'code-reviewer', 'SKILL.md'))
      .addPackage(root)
      .addPackages([])
      .sourceMode('replace')
      .workspace(root)
      .extraAllowedDirectory(extra)
      .extraAllowedDirectories([])
      .toolName('Skill')
      .toolDescriptionTemplate('skills\n%s')
      .onActivate((s) => {
        activated = s.name;
      })
      .glob(true)
      .grep(true)
      .list(false)
      .write(false)
      .shell(false)
      .shellTimeoutMs(1000)
      .confirmShell(() => true)
      .todos(false)
      .askUser(() => ({ q: 'a' }))
      .web({ fetch: true, search: true, braveApiKey: 'k' })
      .memories(true)
      .task({ chatModel: new ScriptedChatModel([{ respond: 'ok' }]), system: 'sub' })
      .chatModel(new ScriptedChatModel([{ respond: 'ok' }]))
      .perSkillSandbox(false)
      .buildTools();
    expect(tools.some((t) => t.toolDefinition.name === 'Skill')).toBe(true);
    expect(tools.some((t) => t.toolDefinition.name === 'WebFetch')).toBe(true);
    expect(tools.some((t) => t.toolDefinition.name === 'WebSearch')).toBe(true);
    expect(tools.some((t) => t.toolDefinition.name === 'MemoryView')).toBe(true);
    expect(tools.some((t) => t.toolDefinition.name === 'Task')).toBe(true);
    expect(tools.some((t) => t.toolDefinition.name === 'AskUserQuestion')).toBe(true);
    const withTaskTrue = SkillsToolbox.builder()
      .sourceMode('replace')
      .addSkill(
        agentSkill({
          name: 'code-reviewer',
          description: 'Reviews TypeScript when asked to review or audit code.',
          content: 'x',
        }),
      )
      .workspace(root)
      .chatModel(new ScriptedChatModel([{ respond: 'ok' }]))
      .task(true)
      .build();
    expect(withTaskTrue.tools.some((t) => t.toolDefinition.name === 'Task')).toBe(true);
    const skill = tools.find((t) => t.toolDefinition.name === 'Skill');
    void skill?.call(JSON.stringify({ command: 'code-reviewer' }));
    expect(activated).toBe('code-reviewer');
  });

  test('SkillsAgent.builder extras, of(), and memories system prompt', () => {
    const model = new ScriptedChatModel([{ respond: 'hi' }]);
    const extra = functionToolCallback({
      name: 'ping',
      description: 'ping',
      inputSchema: { type: 'object', properties: {} },
      call: () => 'pong',
    });
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-agent2-'));
    const agent = SkillsAgent.builder()
      .chatModel(model)
      .system('sys')
      .extraTools(extra)
      .chatClient(ChatClient.create(model))
      .defaultOptions({ temperature: 0 })
      .advisors()
      .defaultConversationId('c1')
      .conversationMemory({} as never)
      .clientBuilderOptions({})
      .sourceMode('replace')
      .addSkill(
        agentSkill({
          name: 'code-reviewer',
          description: 'Reviews TypeScript when asked to review.',
          content: 'x',
        }),
      )
      .workspace(root)
      .memories({ directory: join(root, 'mem') })
      .build();
    expect(agent).toBeDefined();

    const ofAgent = SkillsAgent.of({
      chatModel: model,
      directories: [],
      skills: [
        agentSkill({
          name: 'code-reviewer',
          description: 'Reviews TypeScript when asked to review.',
          content: 'x',
        }),
      ],
      workspace: root,
      memories: true,
    });
    expect(ofAgent).toBeDefined();
    expect(
      createSkillsAgent({
        chatModel: model,
        directories: [],
        skills: [
          agentSkill({
            name: 'code-reviewer',
            description: 'Reviews TypeScript when asked to review.',
            content: 'x',
          }),
        ],
        workspace: root,
      }),
    ).toBeDefined();
    try {
      SkillsToolbox.of();
    } catch {
      // default discovery may find no skills
    }
  });

  test('SkillsTool.builder onActivate and runtime activeSkill', () => {
    let name = '';
    const tool = SkillsTool.builder()
      .addSkill(
        agentSkill({
          name: 'code-reviewer',
          description: 'Reviews TypeScript when asked to review.',
          content: 'body',
        }),
      )
      .onActivate((s) => {
        name = s.name;
      })
      .build();
    void tool.call(JSON.stringify({ command: 'code-reviewer' }));
    expect(name).toBe('code-reviewer');

    const runtime = createSkillsRuntime({
      workspace: '/tmp',
      skillDirectories: [],
    });
    expect(runtime.activeSkill()).toBeUndefined();
    runtime.activate(
      agentSkill({
        name: 'code-reviewer',
        description: 'Reviews TypeScript when asked to review.',
        content: 'x',
      }),
    );
    expect(runtime.activeSkill()?.name).toBe('code-reviewer');
  });

  test('validateSkill throws formatted name errors', () => {
    expect(() =>
      validateSkill(
        agentSkill({
          name: 'Bad_Name',
          description: 'Reviews TypeScript when asked to review.',
          content: 'x',
        }),
        { matchDirectoryName: false },
      ),
    ).toThrow(/Invalid skill/);
  });
});

describe('file and web tools remaining branches', () => {
  test('webFetch and webSearch cover success and errors', async () => {
    const fetchTool = webFetchTool({ timeoutMs: 50, maxChars: 4 });
    expect(await fetchTool.call(JSON.stringify({}))).toContain('url is required');
    expect(await fetchTool.call(JSON.stringify({ url: 'not a url' }))).toContain('Invalid URL');
    expect(await fetchTool.call(JSON.stringify({ url: 'file:///etc/passwd' }))).toContain(
      'Only http',
    );

    const original = globalThis.fetch;
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const parsed =
        input instanceof URL ? input : new URL(typeof input === 'string' ? input : input.url);
      if (parsed.pathname.includes('throw')) throw new Error('net');
      if (parsed.hostname === 'api.search.brave.com') {
        if (parsed.searchParams.get('q') === 'failsearch') {
          return new Response('noop', { status: 500 });
        }
        return new Response(
          JSON.stringify({
            web: { results: [{ title: 'T', url: 'https://ex', description: 'd' }] },
          }),
          { status: 200 },
        );
      }
      return new Response('hello world', { status: 200 });
    }) as unknown as typeof fetch;
    try {
      expect(await fetchTool.call(JSON.stringify({ url: 'https://example.com' }))).toContain(
        'hell',
      );
      expect(await fetchTool.call(JSON.stringify({ url: 'https://example.com/throw' }))).toContain(
        'Error fetching',
      );
      const search = webSearchTool({ apiKey: 'k', timeoutMs: 50 });
      expect(await search.call(JSON.stringify({}))).toContain('query is required');
      expect(await webSearchTool({}).call(JSON.stringify({ query: 'q' }))).toContain(
        'Brave API key missing',
      );
      expect(await search.call(JSON.stringify({ query: 'q', count: 2 }))).toContain('1. T');
      expect(await search.call(JSON.stringify({ query: 'failsearch' }))).toContain(
        'Brave search failed',
      );
      globalThis.fetch = (async () => {
        throw 'string-err';
      }) as unknown as typeof fetch;
      expect(await search.call(JSON.stringify({ query: 'q' }))).toContain('Error searching');
      expect(await webSearchTool({ apiKey: 'k' }).call(JSON.stringify({ query: 'q' }))).toContain(
        'string-err',
      );
    } finally {
      globalThis.fetch = original;
    }

    const fmt = webSearchTool({ apiKey: 'k' });
    globalThis.fetch = (async () =>
      new Response('not-json', { status: 200 })) as unknown as typeof fetch;
    try {
      expect(await fmt.call(JSON.stringify({ query: 'q' }))).toContain('not-json');
      globalThis.fetch = (async () =>
        new Response(JSON.stringify({ web: { results: [] } }), {
          status: 200,
        })) as unknown as typeof fetch;
      expect(await fmt.call(JSON.stringify({ query: 'q' }))).toBe('No search results');
    } finally {
      globalThis.fetch = original;
    }
  });

  test('bash confirm reject, stderr, truncate, and start error', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-bash2-'));
    const denied = bashTool({
      allowedDirectories: [root],
      workingDirectory: root,
      confirm: () => false,
    });
    expect(await denied.call(JSON.stringify({ command: 'echo hi' }))).toContain('not approved');

    const noisy = bashTool({
      allowedDirectories: [root],
      workingDirectory: root,
      maxOutputChars: 8,
    });
    const errOut = await noisy.call(JSON.stringify({ command: 'echo err >&2' }));
    expect(errOut).toContain('stderr');
    const long = await noisy.call(JSON.stringify({ command: 'printf %100s ' }));
    expect(long).toContain('truncated');

    const boom = bashTool({ allowedDirectories: [root], workingDirectory: root });
    const { EventEmitter } = await import('node:events');
    const spawnSpy = spyOn(await import('node:child_process'), 'spawn').mockImplementation((() => {
      const std = () =>
        Object.assign(new EventEmitter(), {
          setEncoding: () => undefined,
          on: EventEmitter.prototype.on,
        });
      const child = Object.assign(new EventEmitter(), {
        stdout: std(),
        stderr: std(),
        kill: () => true,
      }) as unknown as ReturnType<typeof import('node:child_process').spawn>;
      queueMicrotask(() => child.emit('error', new Error('spawn failed')));
      return child;
    }) as unknown as typeof import('node:child_process').spawn);
    try {
      expect(await boom.call(JSON.stringify({ command: 'echo hi' }))).toContain(
        'Error starting command',
      );
    } finally {
      spawnSpy.mockRestore();
    }
  });

  test('edit / write / read / list / glob / grep edge paths', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-files-'));
    const file = join(root, 'a.txt');
    writeFileSync(file, 'hello hello\n');
    mkdirSync(join(root, 'dir'));
    writeFileSync(join(root, 'dir', 'b.ts'), 'const x = 1;\n');

    const edit = editTool({ allowedDirectories: [root] });
    expect(
      await edit.call(JSON.stringify({ filePath: file, oldString: 'a', newString: 'a' })),
    ).toContain('must be different');
    expect(
      await edit.call(
        JSON.stringify({ filePath: join(root, 'missing.txt'), oldString: 'a', newString: 'b' }),
      ),
    ).toContain('does not exist');
    expect(
      await edit.call(
        JSON.stringify({ filePath: join(root, 'dir'), oldString: 'a', newString: 'b' }),
      ),
    ).toContain('directory');
    expect(
      await edit.call(JSON.stringify({ filePath: file, oldString: 'zzz', newString: 'b' })),
    ).toContain('was not found');
    expect(
      await edit.call(
        JSON.stringify({ filePath: file, oldString: 'hello', newString: 'hi', replaceAll: true }),
      ),
    ).toContain('updated');

    const readSpy = spyOn(fs, 'readFileSync').mockImplementationOnce(() => {
      throw new Error('read fail');
    });
    expect(
      await edit.call(JSON.stringify({ filePath: file, oldString: 'hi', newString: 'ho' })),
    ).toContain('Error reading file');
    readSpy.mockRestore();

    writeFileSync(file, 'unique-old');
    const writeSpy = spyOn(fs, 'writeFileSync').mockImplementationOnce(() => {
      throw new Error('write fail');
    });
    expect(
      await edit.call(
        JSON.stringify({ filePath: file, oldString: 'unique-old', newString: 'unique-new' }),
      ),
    ).toContain('Error writing file');
    writeSpy.mockRestore();

    const write = writeTool({ allowedDirectories: [root] });
    expect(await write.call(JSON.stringify({ filePath: join(root, 'dir') }))).toContain(
      'directory',
    );
    const writeFail = spyOn(fs, 'writeFileSync').mockImplementationOnce(() => {
      throw 'noop';
    });
    expect(
      await write.call(JSON.stringify({ filePath: join(root, 'c.txt'), content: 'x' })),
    ).toContain('Error writing file');
    writeFail.mockRestore();

    let writeCalls = 0;
    const overwriteFail = spyOn(fs, 'writeFileSync').mockImplementation((() => {
      writeCalls += 1;
      if (writeCalls === 1) {
        const exists = new Error('exists') as NodeJS.ErrnoException;
        exists.code = 'EEXIST';
        throw exists;
      }
      throw new Error('overwrite fail');
    }) as typeof fs.writeFileSync);
    expect(
      await write.call(JSON.stringify({ filePath: join(root, 'd.txt'), content: 'y' })),
    ).toContain('Error writing file');
    overwriteFail.mockRestore();

    const read = readTool({ allowedDirectories: [root] });
    const readFail = spyOn(fs, 'readFileSync').mockImplementationOnce(() => {
      throw new Error('read fail');
    });
    expect(await read.call(JSON.stringify({ filePath: file }))).toContain('Error reading file');
    readFail.mockRestore();
    expect(await read.call(JSON.stringify({ filePath: file, offset: 999 }))).toContain(
      'No lines to read',
    );

    const list = listDirectoryTool({ allowedDirectories: [root], workingDirectory: root });
    expect(await list.call(JSON.stringify({ path: file }))).toContain('not a directory');
    expect(await list.call(JSON.stringify({ path: join(root, 'no-dir') }))).toContain(
      'does not exist',
    );
    const empty = join(root, 'empty');
    mkdirSync(empty);
    expect(await list.call(JSON.stringify({ path: empty }))).toContain('Empty directory');
    const listFail = spyOn(fs, 'readdirSync').mockImplementationOnce(() => {
      throw new Error('list fail');
    });
    expect(await list.call(JSON.stringify({ path: empty }))).toContain('Error listing directory');
    listFail.mockRestore();

    const glob = globTool({ allowedDirectories: [root], workingDirectory: root });
    expect(await glob.call(JSON.stringify({ pattern: '*', path: file }))).toContain(
      'not a directory',
    );
    expect(
      await glob.call(JSON.stringify({ pattern: '*', path: join(root, 'missing-dir') })),
    ).toContain('does not exist');
    expect(compileGlob('**foo')('xxfoo')).toBe(true);
    expect(compileGlob('a?c')('abc')).toBe(true);

    const grep = grepTool({ allowedDirectories: [root], workingDirectory: root });
    expect(await grep.call(JSON.stringify({ pattern: '(' }))).toContain(
      'Invalid regular expression',
    );
    expect(await grep.call(JSON.stringify({ pattern: 'x', path: join(root, 'noop') }))).toContain(
      'does not exist',
    );
    expect(await grep.call(JSON.stringify({ pattern: 'hello', path: file }))).toContain('hello');
    expect(
      await grep.call(JSON.stringify({ pattern: 'const', glob: '**/*.md', path: root })),
    ).toContain('No matches');
    expect(
      await grep.call(JSON.stringify({ pattern: 'hello', glob: 'no-such-*.xyz', path: file })),
    ).toContain('No matches');
    const fifo = join(root, 'pipe');
    try {
      const { execSync } = await import('node:child_process');
      execSync(`mkfifo ${JSON.stringify(fifo)}`);
      expect(await grep.call(JSON.stringify({ pattern: 'x', path: fifo }))).toContain(
        'not a file or directory',
      );
    } catch {
      // mkfifo not available
    }
    const blocked = join(root, 'blocked.txt');
    writeFileSync(blocked, 'const hidden = 1;\n');
    chmodSync(blocked, 0);
    try {
      const out = await grep.call(JSON.stringify({ pattern: 'hidden-unique-xyz', path: root }));
      expect(out).toContain('No matches');
    } finally {
      chmodSync(blocked, 0o644);
    }
  });

  test('memory list, empty tree, and system prompt', async () => {
    const root = mkdtempSync(join(tmpdir(), 'ai-utils-mem2-'));
    const [view, write, edit, del, rename] = memoryTools({ directory: root });
    if (!view || !write || !edit || !del || !rename) throw new Error('missing view');
    expect(await view.call(JSON.stringify({}))).toContain('Empty memory directory');
    writeFileSync(join(root, 'a.md'), 'x');
    mkdirSync(join(root, 'nested'));
    expect(await view.call(JSON.stringify({}))).toContain('a.md');
    expect(await view.call(JSON.stringify({ path: 'nested' }))).toContain('Empty memory directory');
    expect(await view.call(JSON.stringify({ path: 'missing.md' }))).toContain('does not exist');
    expect(
      await edit.call(JSON.stringify({ path: 'missing.md', oldString: 'a', newString: 'b' })),
    ).toContain('does not exist');
    expect(
      await edit.call(JSON.stringify({ path: 'a.md', oldString: 'zzz', newString: 'b' })),
    ).toContain('oldString was not found');
    expect(await del.call(JSON.stringify({ path: 'missing.md' }))).toContain('does not exist');
    expect(await rename.call(JSON.stringify({ from: 'missing.md', to: 'b.md' }))).toContain(
      'does not exist',
    );
    expect(formatMemorySystemPrompt('/mem')).toContain('/mem');
  });
});

describe('assertPathAllowed remaining branches', () => {
  test('missing allowed root, dangling symlink, and fs errors', () => {
    const missingRoot = join(tmpdir(), `ai-utils-missing-${Date.now()}`);
    const nested = join(missingRoot, 'file.txt');
    expect(assertPathAllowed(nested, [missingRoot]).ok).toBe(true);

    const parent = mkdtempSync(join(tmpdir(), 'ai-utils-dangle-'));
    const allowed = join(parent, 'allowed');
    mkdirSync(allowed);
    const link = join(allowed, 'ghost');
    symlinkSync(join(parent, 'no-such-target'), link);
    expect(assertPathAllowed(link, [allowed]).ok).toBe(false);

    const resolveSpy = spyOn(path, 'resolve').mockImplementationOnce(() => {
      throw new Error('bad resolve');
    });
    expect(assertPathAllowed('x', [allowed]).ok).toBe(false);
    resolveSpy.mockRestore();

    const realSpy = spyOn(fs, 'realpathSync').mockImplementationOnce((() => {
      throw 'boom';
    }) as unknown as typeof fs.realpathSync);
    expect(assertPathAllowed(join(allowed, 'a'), [allowed]).ok).toBe(false);
    realSpy.mockRestore();

    const lstatSpy = spyOn(fs, 'lstatSync').mockImplementation(() => {
      throw new Error('gone');
    });
    expect(assertPathAllowed(join(allowed, 'deep', 'x'), [allowed]).ok).toBe(true);
    lstatSpy.mockRestore();
  });
});
