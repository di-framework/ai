# di-ml

Directory-driven ONNX fine-tuning library. It loads a workspace, lints the graph, updates named f32 initializers (or injected LoRA adapters), and writes `dist/`. The command-line front end is the sibling crate [`di-ml-cli`](../cli).

Embedding losses sit after ONNX outputs and stay out of the exported graph. Graph, train, and loss code call the `Accel` trait only. `DI_ML_ACCEL=cpu|metal|auto` selects one backend at process start.

The [di-ml guide](../../README.md) is the user-facing reference for workspace files, `train.toml`, JSONL rows, losses, pooling, and supported ops. `TrainConfig` in `src/workspace.rs` is the field list. `classify` in `src/ops.rs` is the op list.

## Train a workspace

```rust
let result = di_ml::run_workspace("xor-workspace")?;
// result.dist_model → <workspace>/dist/model.onnx
// result.report    → steps, accel, trainable names
```

`run_workspace` requires `model.onnx`, `train.toml`, and `data/train.jsonl`. It rejects unknown ONNX ops before the first step. Only f32 initializers are trained. AdamW state is allocated only for that set.

`workspace::load` returns the parsed `Workspace` without training.

## Run the exported graph

```rust
use std::collections::HashMap;

use di_ml::Session;

let mut session = Session::from_path("xor-workspace/dist/model.onnx")?;
let outputs = session.run_f32(&feeds)?;
```

`Session` is forward-only: no workspace directory, tape, or optimizer. `Session::from_bytes` takes the ONNX blob. `run` accepts mixed `TensorValue` feeds; `run_f32` is the f32 path. Input and output names match the trained graph.

TypeScript inference is `@di-framework/ml` in `packages/ml/infer` (ONNX Runtime Web), not this crate.

## Scaffold

```rust
let root = di_ml::scaffold(&di_ml::InitSpec::xor_demo(path))?;
```

That writes the same XOR demo the CLI `init` command writes when stdin is not a terminal.

## Entry points

Re-exported from `src/lib.rs`:

| Item | Role |
| --- | --- |
| `run_workspace`, `RunResult` | Load a workspace, train, write `dist/` |
| `Session` | Forward-only inference on ONNX bytes |
| `scaffold`, `InitSpec` | Write the XOR demo workspace |
| `load`, `TrainConfig`, `Workspace` | Parse `train.toml` and locate data files |
| `ExecGraph` | Autodiff interpreter |
| `Error` | `usage` → exit 2, `fail` → exit 3 |

Tests live next to the modules they cover (`cargo test -p di-ml`).
