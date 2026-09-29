# di-ml

Fine-tune ONNX inference graphs from directory workspaces.

`di-ml` loads `model.onnx`, runs an autodiff interpreter over a transformer-shaped op subset, updates named f32 initializers (or LoRA adapters), and writes the trained inference model to `dist/`. Losses that need batch geometry (InfoNCE, cosine, distillation) sit *after* the ONNX outputs so exporters do not have to bake them in.

This is not ONNX Runtime on-device training and not a PyTorch replacement. The product is a workspace-native embedder fine-tuner: same CLI, one Accel backend.

## CLI

```bash
# scaffold an XOR demo workspace (interactive on a TTY; XOR defaults otherwise)
di-ml init [dir]

# train: exactly one positional argument, the workspace directory
di-ml <workspace>
```

Exit status: `0` success, `2` usage, `3` failure.

```bash
cargo run -p di-ml-cli -- init xor-workspace
cargo run -p di-ml-cli -- xor-workspace
```

Prebuilt CLIs are prereleases of `@di-framework/ml`. The suffix is the OS and CPU (`6.0.1-darwin-aarch64`, `6.0.1-linux-x64`, `6.0.1-linux-aarch64`, `6.0.1-windows-x64`). `6.0.1` tracks the workspace release. `npm install @di-framework/ml` still installs the TypeScript session on `latest` (`0.1.0`).

```bash
npm install -g @di-framework/ml@6.0.1-darwin-aarch64
```

Linux packages are glibc builds from the GitHub runner.

## Workspace

```
<workspace>/
  model.onnx          # forward ONNX graph
  train.toml          # loss, optimizer, LoRA, pooling, …
  data/train.jsonl    # required
  data/eval.jsonl     # optional
  frozen.txt          # optional, one initializer name (or regex) per line
  trainable.txt       # optional; names or regexes
  dist/model.onnx     # written after train (same I/O names as input)
  dist/metrics.json
  dist/optimizer.json # AdamW moments; reload with resume = true
  dist/adapter.onnx   # only if lora-export = "adapter"
```

`train.toml` uses kebab-case keys. A `[train]` table is also accepted. Snake_case aliases exist on several fields.

```toml
loss = "mse"
optimizer = "adamw"
learning-rate = 0.05
max-steps = 400
batch-size = 4
seed = 1
log-every = 50
```

Embedder / LoRA example:

```toml
[train]
loss = "infonce"
trainable = ["lora_.*", ".*attn.*out_proj.*", "embed_head"]
freeze-embeddings = true
lora-rank = 16
lora-alpha = 32
lora-targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lora-export = "merge"
pool = "last-token"
l2-normalize = true
temperature = 0.07
microbatch-size = 4
grad-accum-steps = 8
activation-checkpointing = true
mixed-precision = "fp16"
eval-metric = "spearman"
tokenizer = "tokenizer.json"
instruction-query = "Instruct: retrieve relevant passages.\nQuery: "
max-length = 512
resume = false
```

Loss: `mse`, `cross-entropy`, `bce-logits`, `l1`, `infonce`, `cosine`, `triplet`, `supcon`, `distill`.  
Optimizer: `adamw`, `sgd`.  
Pool: `none`, `last-token`, `mean`, `cls`.  
Mixed precision: `off`, `fp16`, `bf16` (matmul inputs are quantized; compute stays f32).  
Eval metric: `loss`, `spearman`, `ndcg`.  
LoRA export: `merge` (default), `adapter`.

If `lora-rank` is set and `lora-targets` is omitted, the default target list is `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. `tokenizer` is a Hugging Face `tokenizers` JSON (path relative to the workspace, or absolute). `instruction-query` / `instruction-doc` are optional prefixes. `triplet-margin` and `distill-weight` apply to those losses.

Each JSONL row is an object. Keys that match the model’s runtime inputs are feeds (int64 token ids stay ints). Labels are `label`, `labels`, or `y`. Contrastive rows may add `positive` (token ids, a nested object, or a string), `teacher_pos`, `teacher_neg`, `replay_emb`, `score`, `attention_mask`. With a tokenizer, `query` / `text` / `sentence1` and `positive` / `sentence2` strings are encoded into `input_ids`.

```json
{"input":[0.0,1.0],"label":[1.0]}
{"input_ids":[1,2,3],"positive":[4,5,6]}
{"query":"what is rust?","positive":"a systems language"}
```

For `cross-entropy`, a scalar label is a class index. Spearman eval reads `score` on `data/eval.jsonl` pairs.

Unknown ONNX ops are rejected *before* epoch 0 with a missing-op list (the roadmap).

## GPU

Training picks a compute backend at process start and uses it for GEMM, batched GEMM, elementwise, softmax, and fused SDPA. Graph and optimizer code do not branch on GPU APIs.

| Host | Backend |
| --- | --- |
| macOS with a Metal GPU (M-series, including M4 Max) | Metal (unified memory, compute shaders, fused SDPA) |
| anything else | CPU (`ndarray`) |

Small GEMMs and elementwise ops stay on CPU even when Metal is selected; GPU launch cost would dominate. Larger matmuls and attention run on the GPU.

Override with `DI_ML_ACCEL=cpu` or `DI_ML_ACCEL=metal`. `auto` (default) enables GPU when the OS gate and a device are present. CUDA / ROCm / DirectML / MLX slots are reserved behind the same `Accel` trait.

`dist/metrics.json` records `accel`, `peak-bytes`, `step-time-ms`, `seed`, `config-hash`, `data-hash`, `mixed-precision`, and (when selected) `eval-spearman` / `eval-ndcg`.

## Supported ops

**Ready:** Gemm, MatMul (2D–4D), Add/Sub/Mul/Div, Relu, Sigmoid, Tanh, Softplus, Softmax, Identity, Dropout, Cast, Neg, Flatten, Reshape (`-1`), Transpose (any perm), Constant, ReduceSum/Mean/Max, Where, Gather / GatherElements / Embedding, Expand, Unsqueeze/Squeeze, Concat, Split, Slice (step=1), LayerNormalization / SkipLayerNormalization, RMSNorm (`SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`), SiLU/Swish, GELU (`BiasGelu`, `FastGelu`, `QuickGelu`), Pow, Sqrt, Clip, Erf, Exp, Log, Abs, Reciprocal, Sin, Cos, Equal/Greater/GreaterOrEqual/Less/LessOrEqual, Not/And/Or, Shape, RoPE / RotaryEmbedding, ScaledDotProductAttention / SDPA / Attention / GroupQueryAttention / MultiHeadAttention (fused fwd; backward recomputes).

**Roadmap P2 (linted, not trained):** Einsum, Pad, Tile, Range, OneHot.

**Roadmap P3 (linted, not trained):** Conv / ConvTranspose, MaxPool / AveragePool / GlobalAveragePool, BatchNormalization, HardSwish / HardSigmoid, GroupNormalization, InstanceNormalization. The English PP-OCRv4 rec fixture (`examples/ocr`) is this bucket: `@di-framework/ml` runs it; `di-ml` rejects it before epoch 0.

Only f32 initializers are trained. Frozen names skip the optimizer. AdamW state is allocated only for the trainable set.

## LoRA

When `lora-rank` is set, `di-ml` injects `MatMul`+`Add` adapters on named Gemm/MatMul nodes (`lora_A.*`, `lora_B.*`, B initialized at 0). Adapters are scaled by `lora-alpha / lora-rank` (default alpha 32). If `trainable` is empty, only those adapters are updated. Export default is merge into the base Gemm (`dist/model.onnx` keeps the original I/O names). `lora-export = "adapter"` writes `dist/adapter.onnx` plus the untouched base graph.

## Init demo

`di-ml init` writes a 2→8→1 MLP (Gemm + Relu + Gemm + Sigmoid), `train.toml`, and the four XOR rows in both `data/train.jsonl` and `data/eval.jsonl`. Non-interactive stdin uses those defaults. The TTY prompt offers a subset of losses (`mse`, `cross-entropy`, `bce-logits`, `l1`, `infonce`, `cosine`); the rest are set in `train.toml`.

## Inference

`dist/model.onnx` is a forward ONNX graph with the same input and output names as the workspace `model.onnx`. Training losses, LoRA nodes, and optimizer state are not in that file (unless `lora-export = "adapter"`, which writes adapters separately).

The Rust library loads those bytes as a forward-only session — no workspace directory, no tape, no optimizer:

```rust
use di_ml::Session;

let mut session = Session::from_path("workspace/dist/model.onnx")?;
let outputs = session.run_f32(&feeds)?;
```

TypeScript (`packages/ml/infer`, `@di-framework/ml`) does the same with ONNX Runtime Web (WASM, optional WebGPU):

```ts
import { Session } from "@di-framework/ml";

const session = await Session.fromPath("workspace/dist/model.onnx");
const out = await session.run({
  input: { data: new Float32Array([0, 1]), dims: [1, 2] },
});
```

`fromBytes` takes the ONNX blob when there is no filesystem.

OCR (English line recognition) uses the same `Session`. Fetch the pinned PP-OCRv4 rec ONNX, then:

```ts
import { Session, loadCharset, recognizeLine, renderGlyphLine } from "@di-framework/ml";

const session = await Session.fromPath("examples/ocr/models/en_PP-OCRv4_rec_mobile.onnx");
const charset = loadCharset(await Bun.file("examples/ocr/en_dict.txt").text());
const { text } = await recognizeLine(session, renderGlyphLine("HELLO"), charset);
```

```bash
bun packages/ml/scripts/fetch-ocr-model.ts
```

`di-ml` will not train that graph until Conv and the rest of the vision subset are Ready. Provenance: `packages/ml/examples/ocr/SOURCE.md`.

```bash
bun packages/ml/scripts/eval.ts
```

That writes `packages/ml/docs/eval/latest.md`: pass / partial / unproven / stub / missing. Use it to see where the work is.
