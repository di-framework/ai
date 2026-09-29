#!/usr/bin/env bun
/**
 * Capability scorecard. Reads docs/eval/tasks.json, runs cargo/bun probes,
 * writes docs/eval/latest.md. See docs/eval.md.
 */
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const REPO = join(ROOT, '../..');
const TASKS_PATH = join(ROOT, 'docs/eval/tasks.json');
const OUT_PATH = join(ROOT, 'docs/eval/latest.md');

type Code = 'ready' | 'stub' | 'missing';
type Coverage = 'e2e' | 'unit' | 'none';
type Status = 'pass' | 'partial' | 'unproven' | 'fail' | 'stub' | 'missing' | 'skip';

type Probe = { kind: 'cargo' | 'bun'; name: string };

type Task = {
  id: string;
  family: string;
  title: string;
  expect: string;
  done_when: string;
  code: Code;
  coverage: Coverage;
  probes: Probe[];
  roadmap?: string;
  notes?: string;
  when?: 'macos' | 'always';
};

type Catalog = {
  version: number;
  families: { id: string; title: string; expect: string }[];
  tasks: Task[];
};

type ProbeResult = 'ok' | 'failed' | 'ignored' | 'absent';

const args = new Set(process.argv.slice(2));
const catalogOnly = args.has('--catalog');
const withCli = args.has('--cli');

const catalog = (await Bun.file(TASKS_PATH).json()) as Catalog;

function isMac() {
  return process.platform === 'darwin';
}

async function runCaptured(cmd: string[], cwd = ROOT) {
  const proc = Bun.spawn(cmd, {
    cwd,
    stdout: 'pipe',
    stderr: 'pipe',
    env: { ...process.env, CARGO_TERM_COLOR: 'never', NO_COLOR: '1' },
  });
  const [stdout, stderr, exit] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  return { stdout, stderr, combined: `${stdout}\n${stderr}`, exit };
}

function parseCargo(text: string): Map<string, ProbeResult> {
  const out = new Map<string, ProbeResult>();
  const re = /^test (\S+) \.\.\. (ok|FAILED|ignored|ignored,.*)$/gm;
  for (const m of text.matchAll(re)) {
    const full = m[1];
    const raw = m[2];
    if (!full || !raw) continue;
    const status = raw.startsWith('ignored') ? 'ignored' : raw === 'FAILED' ? 'failed' : 'ok';
    out.set(full, status);
    const short = full.split('::').pop();
    if (short) out.set(short, status);
  }
  return out;
}

function parseBun(text: string): Map<string, ProbeResult> {
  const out = new Map<string, ProbeResult>();
  const re = /^\((pass|fail|skip)\)\s+(.+)$/gm;
  for (const m of text.matchAll(re)) {
    const kind = m[1];
    const title = m[2];
    if (!kind || !title) continue;
    const status: ProbeResult = kind === 'pass' ? 'ok' : kind === 'fail' ? 'failed' : 'ignored';
    out.set(title.trim(), status);
  }
  return out;
}

function lookup(
  probe: Probe,
  cargo: Map<string, ProbeResult>,
  bun: Map<string, ProbeResult>,
): ProbeResult {
  if (probe.kind === 'cargo') {
    return cargo.get(probe.name) ?? 'absent';
  }
  for (const [title, status] of bun) {
    if (title.includes(probe.name)) return status;
  }
  return 'absent';
}

function score(
  task: Task,
  cargo: Map<string, ProbeResult>,
  bun: Map<string, ProbeResult>,
): { status: Status; detail: string } {
  const gated = task.when === 'macos' && !isMac();
  if (task.code === 'missing') {
    return { status: 'missing', detail: task.roadmap ?? 'not built' };
  }
  if (task.code === 'stub') {
    return { status: 'stub', detail: task.roadmap ?? 'toml/API lies' };
  }
  if (task.probes.length === 0) {
    return { status: 'unproven', detail: 'no probe' };
  }
  if (gated) {
    return { status: 'skip', detail: 'macos-only' };
  }

  const results = task.probes.map((p) => ({
    p,
    r: lookup(p, cargo, bun),
  }));
  const failed = results.filter((x) => x.r === 'failed');
  if (failed.length) {
    return {
      status: 'fail',
      detail: failed.map((x) => x.p.name).join(', '),
    };
  }
  const absent = results.filter((x) => x.r === 'absent');
  if (absent.length) {
    if (task.when === 'macos') {
      return { status: 'skip', detail: 'not compiled on this host' };
    }
    return {
      status: 'unproven',
      detail: `probe not in run: ${absent.map((x) => x.p.name).join(', ')}`,
    };
  }
  if (results.every((x) => x.r === 'ignored')) {
    return { status: 'skip', detail: 'ignored' };
  }
  if (task.coverage === 'e2e') {
    return { status: 'pass', detail: results.map((x) => x.p.name).join(', ') };
  }
  return {
    status: 'partial',
    detail: `${task.coverage}: ${results.map((x) => x.p.name).join(', ')}`,
  };
}

function render(
  catalog: Catalog,
  rows: { task: Task; status: Status; detail: string }[],
  meta: { cargoExit: number | null; bunExit: number | null; cli: string | null; elapsedMs: number },
): string {
  const counts = new Map<Status, number>();
  for (const r of rows) counts.set(r.status, (counts.get(r.status) ?? 0) + 1);
  const order: Status[] = ['fail', 'stub', 'unproven', 'missing', 'partial', 'pass', 'skip'];
  const summary = order
    .filter((s) => counts.get(s))
    .map((s) => `${counts.get(s)} ${s}`)
    .join(', ');

  const work = rows.filter((r) => ['fail', 'stub', 'unproven', 'missing'].includes(r.status));

  let md = `# Capability scorecard\n\n`;
  md += `Generated by \`bun packages/ml/scripts/eval.ts\`. Do not edit by hand. Catalog: \`packages/ml/docs/eval/tasks.json\`. Process: \`packages/ml/docs/eval.md\`.\n\n`;
  md += `- Host: ${process.platform} ${process.arch}\n`;
  md += `- Accel: \`${process.env.DI_ML_ACCEL ?? 'auto'}\`\n`;
  md += `- cargo test exit: ${meta.cargoExit ?? 'n/a'}; bun test exit: ${meta.bunExit ?? 'n/a'}\n`;
  if (meta.cli) md += `- CLI smoke: ${meta.cli}\n`;
  md += `- Elapsed: ${(meta.elapsedMs / 1000).toFixed(1)}s\n`;
  md += `- **${summary}**\n\n`;

  md += `## Where the work is\n\n`;
  if (work.length === 0) {
    md += `No fail/stub/unproven/missing tasks.\n\n`;
  } else {
    md += `| Status | ID | Task | Next |\n| --- | --- | --- | --- |\n`;
    for (const r of work) {
      const next =
        r.status === 'fail'
          ? `fix probe (${r.detail})`
          : r.status === 'stub'
            ? `implement or delete (${r.detail})`
            : r.status === 'unproven'
              ? `add a probe (${r.detail})`
              : `build ${r.task.roadmap ?? r.detail}`;
      md += `| ${r.status} | ${r.task.id} | ${r.task.title} | ${next} |\n`;
    }
    md += `\n`;
  }

  for (const fam of catalog.families) {
    const subset = rows.filter((r) => r.task.family === fam.id);
    const famCounts = order
      .filter((s) => subset.some((r) => r.status === s))
      .map((s) => `${subset.filter((r) => r.status === s).length} ${s}`)
      .join(', ');
    md += `## ${fam.title}\n\n${fam.expect}\n\n`;
    md += `_${famCounts}_\n\n`;
    md += `| ID | Task | Status | Evidence |\n| --- | --- | --- | --- |\n`;
    for (const r of subset) {
      md += `| ${r.task.id} | ${r.task.title} | ${r.status} | ${r.detail.replace(/\|/g, '/')} |\n`;
    }
    md += `\n`;
  }
  return md;
}

if (catalogOnly) {
  console.log(`# ${catalog.tasks.length} tasks in ${catalog.families.length} families\n`);
  for (const fam of catalog.families) {
    console.log(`## ${fam.title}`);
    for (const t of catalog.tasks.filter((x) => x.family === fam.id)) {
      const n = t.probes.length ? t.probes.map((p) => p.name).join(', ') : 'no probe';
      console.log(`- ${t.id} ${t.title}  [${t.code}/${t.coverage}]  ${n}`);
    }
    console.log('');
  }
  process.exit(0);
}

const t0 = Date.now();
try {
  const { ensureOcrModel } = await import('./fetch-ocr-model.ts');
  console.error('ensuring OCR rec fixture …');
  await ensureOcrModel();
} catch (err) {
  console.error(`OCR fixture fetch failed (lint test will skip): ${err}`);
}
console.error('running cargo test --workspace …');
const cargoRun = await runCaptured(['cargo', 'test', '--workspace'], REPO);
const cargo = parseCargo(cargoRun.combined);
if (cargo.size === 0) {
  console.error('cargo test produced no libtest lines:');
  console.error(cargoRun.combined.slice(-4000));
}

console.error('running bun test (packages/ml/infer) …');
const bunRun = await runCaptured(['bun', 'test'], join(ROOT, 'infer'));
const bun = parseBun(bunRun.combined);

let cliNote: string | null = null;
if (withCli) {
  const dir = mkdtempSync(join(tmpdir(), 'di-ml-eval-'));
  try {
    const init = await runCaptured(
      ['cargo', 'run', '-q', '-p', 'di-ml-cli', '--', 'init', dir],
      REPO,
    );
    const train = await runCaptured(['cargo', 'run', '-q', '-p', 'di-ml-cli', '--', dir], REPO);
    const ok = init.exit === 0 && train.exit === 0 && train.combined.includes('wrote');
    cliNote = ok ? `ok (${dir})` : `fail init=${init.exit} train=${train.exit}`;
    if (!ok) {
      console.error(init.combined.slice(-1500));
      console.error(train.combined.slice(-1500));
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

const rows = catalog.tasks.map((task) => {
  const { status, detail } = score(task, cargo, bun);
  return { task, status, detail };
});

if (withCli && cliNote?.startsWith('fail')) {
  const ws = rows.find((r) => r.task.id === 'T-WS-02');
  if (ws) {
    ws.status = 'fail';
    ws.detail = `cli smoke: ${cliNote}`;
  }
}

const md = render(catalog, rows, {
  cargoExit: cargoRun.exit,
  bunExit: bunRun.exit,
  cli: cliNote,
  elapsedMs: Date.now() - t0,
});

mkdirSync(dirname(OUT_PATH), { recursive: true });
writeFileSync(OUT_PATH, md);

const counts = new Map<Status, number>();
for (const r of rows) counts.set(r.status, (counts.get(r.status) ?? 0) + 1);

console.log(md.split('## Workspace CLI')[0]?.trim() ?? '');
console.log(`\nWrote ${OUT_PATH}`);

const failed = rows.filter((r) => r.status === 'fail');
const cargoFails = [...cargo.entries()].filter(([k, v]) => v === 'failed' && !k.includes('::'));
const bunFails = [...bun.entries()].filter(([, v]) => v === 'failed');
if (failed.length || cargo.size === 0 || cargoFails.length || bunFails.length) {
  if (cargoFails.length) {
    console.error(`unmapped cargo failures: ${cargoFails.map(([k]) => k).join(', ')}`);
  }
  if (bunFails.length) {
    console.error(`bun failures: ${bunFails.map(([k]) => k).join(', ')}`);
  }
  console.error(`eval failed: ${failed.map((f) => f.task.id).join(', ') || 'see cargo/bun'}`);
  process.exit(1);
}
process.exit(0);
