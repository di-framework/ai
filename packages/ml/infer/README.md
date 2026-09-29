# @di-framework/ml

Load a `di-ml` `dist/model.onnx` and run forward inference. WASM CPU by default; pass WebGPU when the host has it.

```ts
import { Session } from "@di-framework/ml";

const session = await Session.fromPath("workspace/dist/model.onnx");
const out = await session.run({
  input: { data: new Float32Array([0, 1]), dims: [1, 2] },
});
```

`Session.fromBytes` takes the ONNX blob when there is no filesystem.

```ts
await Session.fromBytes(bytes, { executionProviders: ["webgpu", "wasm"] });
```

Tensor types follow their data arrays: `Float32Array` → float32, `Int32Array` →
int32, and `BigInt64Array` → int64. Outputs preserve those types; narrow
`output.data` before doing float-only work. Other tensor types are rejected.

```ts
const out = await session.run({
  input_ids: { data: new BigInt64Array([50281n, 123n, 50282n]), dims: [1, 3] },
  attention_mask: { data: new BigInt64Array([1n, 1n, 1n]), dims: [1, 3] },
});
// Use the input/output names and shapes defined by your graph.
await session.release(); // after the final run; repeated release calls are safe
```

Reuse a session across requests. Tokenization, model-specific pooling, and
normalization belong in the caller. The sibling `example-agents/agents/legal`
integration runs a pinned Free Law Project ModernBERT ONNX this way, with a
Python-versus-WASM parity check before accepting the generated bundle.

Train with the Rust CLI; this package does not train. Prebuilt `di-ml` binaries ship as prereleases of this name (`@di-framework/ml@6.0.1-darwin-aarch64` and the other OS/arch suffixes). `npm install @di-framework/ml` installs this session.

OCR line recognition (PP-OCRv4 English rec) is the first non-XOR fixture:

```ts
import { Session, loadCharset, recognizeLine, renderGlyphLine } from "@di-framework/ml";

const session = await Session.fromPath("../examples/ocr/models/en_PP-OCRv4_rec_mobile.onnx");
const charset = loadCharset(await Bun.file("../examples/ocr/en_dict.txt").text());
const { text } = await recognizeLine(session, renderGlyphLine("HELLO"), charset);
```

```bash
bun packages/ml/scripts/fetch-ocr-model.ts   # from the di-framework/ai repo root
bun test                                 # in packages/ml/infer
```

`bun test` generates the tiny XOR and integer ONNX fixtures with Bun before running
Session tests. Their definitions live in `tests/fixtures.ts`; the XOR weights are
fixed test weights, not a training result. OCR setup generates `en_dict.txt` in the
model's character order and downloads the pinned OCR graph when absent. The first
OCR test needs network access; later runs use the checked local model. These
generated files are excluded from Git. No Python or Rust build is needed for the
TypeScript tests.

The ONNX is fetched (sha256 pin in `examples/ocr/SOURCE.md`), not committed. `di-ml` lints it as P3 vision and will not train it.
