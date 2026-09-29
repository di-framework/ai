import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { missingPackedEntries, packPaths } from './publish-workspace.ts';

/**
 * Prebuilt `di-ml` CLIs publish as prereleases of `@di-framework/ml`.
 * The suffix is `<os>-<arch>` (`darwin-aarch64`). Semver rejects `_`, so `x86_64` is spelled `x64`.
 * `latest` stays the TypeScript session; each publish uses `--tag <suffix>`.
 */
const PACKAGE_NAME = '@di-framework/ml';

export type NativePlatform = {
  /** Rust host triple from `rustc -vV`. */
  target: string;
  /** Semver prerelease identifier appended to MAJOR.MINOR.PATCH. */
  suffix: string;
  /** npm `os` field. */
  os: string;
  /** npm `cpu` field. Node reports Apple Silicon as `arm64`. */
  cpu: string;
  binFile: string;
};

const PLATFORMS: NativePlatform[] = [
  {
    target: 'aarch64-apple-darwin',
    suffix: 'darwin-aarch64',
    os: 'darwin',
    cpu: 'arm64',
    binFile: 'di-ml',
  },
  {
    target: 'x86_64-apple-darwin',
    suffix: 'darwin-x64',
    os: 'darwin',
    cpu: 'x64',
    binFile: 'di-ml',
  },
  {
    target: 'x86_64-unknown-linux-gnu',
    suffix: 'linux-x64',
    os: 'linux',
    cpu: 'x64',
    binFile: 'di-ml',
  },
  {
    target: 'aarch64-unknown-linux-gnu',
    suffix: 'linux-aarch64',
    os: 'linux',
    cpu: 'arm64',
    binFile: 'di-ml',
  },
  {
    target: 'x86_64-pc-windows-msvc',
    suffix: 'windows-x64',
    os: 'win32',
    cpu: 'x64',
    binFile: 'di-ml.exe',
  },
  {
    target: 'aarch64-pc-windows-msvc',
    suffix: 'windows-aarch64',
    os: 'win32',
    cpu: 'arm64',
    binFile: 'di-ml.exe',
  },
];

type NativeManifest = {
  name: string;
  version: string;
  description: string;
  bin: Record<string, string>;
  os: string[];
  cpu: string[];
  files: string[];
  license: string;
  repository: { type: string; url: string; directory: string };
};

export function platformForTarget(target: string): NativePlatform {
  const found = PLATFORMS.find((platform) => platform.target === target);
  if (!found) {
    throw new Error(
      `unsupported rustc host ${target}. Supported: ${PLATFORMS.map((platform) => platform.target).join(', ')}`,
    );
  }
  return found;
}

export function parseRustcHost(verboseVersion: string): string {
  const line = verboseVersion.split('\n').find((entry) => entry.startsWith('host: '));
  if (!line) throw new Error('rustc -vV did not report a host');
  return line.slice('host: '.length).trim();
}

/** `6.0.1` + `darwin-aarch64` → `6.0.1-darwin-aarch64`. */
export function nativeVersion(base: string, suffix: string): string {
  if (!/^\d+\.\d+\.\d+$/.test(base)) {
    throw new Error(`base version must be MAJOR.MINOR.PATCH, got ${base}`);
  }
  if (!/^[0-9A-Za-z-]+$/.test(suffix)) {
    throw new Error(`suffix ${suffix} is not a semver prerelease identifier`);
  }
  return `${base}-${suffix}`;
}

export function assetFileName(platform: NativePlatform): string {
  const ext = platform.binFile.endsWith('.exe') ? '.exe' : '';
  return `di-ml-${platform.suffix}${ext}`;
}

export function stageNativePackage(options: {
  stageDir: string;
  rootDir: string;
  baseVersion: string;
  platform: NativePlatform;
  binaryPath: string;
}): NativeManifest {
  const { stageDir, rootDir, baseVersion, platform, binaryPath } = options;
  if (!existsSync(binaryPath)) throw new Error(`release binary missing: ${binaryPath}`);
  const binDir = join(stageDir, 'bin');
  mkdirSync(binDir, { recursive: true });
  const stagedBin = join(binDir, platform.binFile);
  copyFileSync(binaryPath, stagedBin);
  chmodSync(stagedBin, 0o755);

  const files = ['bin', 'README.md'];
  for (const name of ['LICENSE', 'LICENSE-MIT', 'LICENSE-APACHE']) {
    const source = join(rootDir, name);
    if (!existsSync(source)) continue;
    copyFileSync(source, join(stageDir, name));
    files.push(name);
  }

  const version = nativeVersion(baseVersion, platform.suffix);
  const manifest: NativeManifest = {
    name: PACKAGE_NAME,
    version,
    description: `Prebuilt di-ml CLI (${platform.suffix})`,
    bin: { 'di-ml': `bin/${platform.binFile}` },
    os: [platform.os],
    cpu: [platform.cpu],
    files,
    license: '(MIT OR Apache-2.0)',
    repository: {
      type: 'git',
      url: 'https://github.com/di-framework/ai',
      directory: 'packages/ml/crates/cli',
    },
  };
  writeFileSync(join(stageDir, 'package.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  writeFileSync(join(stageDir, 'README.md'), nativeReadme(version, platform));
  return manifest;
}

function nativeReadme(version: string, platform: NativePlatform): string {
  return `# @di-framework/ml@${version}

Prebuilt \`di-ml\` CLI for ${platform.os} (${platform.cpu}).

\`\`\`bash
npm install -g @di-framework/ml@${version}
di-ml init
\`\`\`

This version is the native trainer. The TypeScript ONNX Runtime Web session stays on the \`latest\` tag.
`;
}

function readBaseVersion(rootDir: string): string {
  const root = JSON.parse(readFileSync(join(rootDir, 'package.json'), 'utf8')) as {
    version?: string;
  };
  if (!root.version) throw new Error('root package.json has no version');
  return root.version;
}

function rustcHost(): string {
  const result = Bun.spawnSync(['rustc', '-vV'], { stdout: 'pipe', stderr: 'pipe' });
  if (result.exitCode !== 0) {
    throw new Error(result.stderr.toString() || 'rustc -vV failed');
  }
  return parseRustcHost(result.stdout.toString());
}

function buildRelease(rootDir: string): void {
  const build = Bun.spawnSync(['cargo', 'build', '--release', '-p', 'di-ml-cli'], {
    cwd: rootDir,
    stdout: 'inherit',
    stderr: 'inherit',
  });
  if (build.exitCode !== 0) throw new Error('cargo build --release -p di-ml-cli failed');
}

function alreadyPublished(name: string, version: string): boolean {
  const result = Bun.spawnSync(['bun', 'pm', 'view', `${name}@${version}`, 'version'], {
    stdout: 'pipe',
    stderr: 'pipe',
  });
  return result.exitCode === 0;
}

export function publishNativeCli(rootDir = process.cwd(), dryRun = false): string {
  const baseVersion = readBaseVersion(rootDir);
  const platform = platformForTarget(rustcHost());
  buildRelease(rootDir);
  const binaryPath = join(rootDir, 'target', 'release', platform.binFile);
  const stageDir = mkdtempSync(join(tmpdir(), 'di-ml-native-'));
  try {
    const manifest = stageNativePackage({
      stageDir,
      rootDir,
      baseVersion,
      platform,
      binaryPath,
    });
    const packed = packPaths(stageDir);
    const missing = missingPackedEntries(manifest.files, packed);
    if (missing.length > 0) {
      throw new Error(
        `${manifest.name}@${manifest.version} pack is missing ${missing.join(', ')}. Refusing to publish.`,
      );
    }
    const assetDir = join(rootDir, 'target', 'native-dist');
    mkdirSync(assetDir, { recursive: true });
    const assetPath = join(assetDir, assetFileName(platform));
    copyFileSync(binaryPath, assetPath);
    chmodSync(assetPath, 0o755);
    const label = `${manifest.name}@${manifest.version}`;
    if (dryRun) {
      console.log(`Dry run ${label} (${packed.length} files) -> ${assetPath}`);
      return label;
    }
    if (alreadyPublished(manifest.name, manifest.version)) {
      console.log(`Skipping ${label} (already published)`);
      return label;
    }
    console.log(`Publishing ${label} (--tag ${platform.suffix})`);
    const publish = Bun.spawnSync(
      [
        'npm',
        'publish',
        '--access',
        'public',
        '--provenance',
        '--ignore-scripts',
        '--tag',
        platform.suffix,
      ],
      { cwd: stageDir, stdout: 'inherit', stderr: 'inherit' },
    );
    if (publish.exitCode !== 0) throw new Error(`npm publish failed for ${label}`);
    console.log(`Asset ${assetPath}`);
    return label;
  } finally {
    rmSync(stageDir, { recursive: true, force: true });
  }
}

if (import.meta.main) {
  try {
    publishNativeCli(process.cwd(), Bun.argv.includes('--dry-run'));
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  }
}
