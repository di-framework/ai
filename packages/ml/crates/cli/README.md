# di-ml-cli

Binary crate for the `di-ml` command. It parses arguments, prompts during `init`, and prints the training report. Graph execution, losses, and file layout live in the [`di-ml`](../di-ml) library.

Package name: `di-ml-cli`. Installed binary name: `di-ml`.

## Commands

```bash
cargo run -p di-ml-cli -- init [dir]    # default dir: xor-workspace
cargo run -p di-ml-cli -- <workspace>
```

After `cargo build -p di-ml-cli`, the same binary is `target/debug/di-ml`.

Release builds are also published as `@di-framework/ml@<version>-<os>-<arch>`, for example `@di-framework/ml@6.0.1-darwin-aarch64`. That prerelease does not replace the `latest` tag.

`init` scaffolds an XOR demo workspace (a 2→8→1 MLP, `train.toml`, and four XOR rows). On a TTY it prompts for the directory when none was passed, then loss, optimizer, learning rate, steps, batch size, and seed. Otherwise it writes the XOR defaults. An existing `model.onnx` is replaced only when that prompt is confirmed.

The only subcommand is `init`. Training is the default command and takes the workspace directory as its only argument. Every knob is a file under that directory.

Exit status: `0` success, `2` usage, `3` failure.

One compute backend is chosen for the process:

```bash
DI_ML_ACCEL=auto    # default: Metal on macOS when a GPU is present, otherwise CPU
DI_ML_ACCEL=cpu
DI_ML_ACCEL=metal   # panics if no Metal device is available
```

Workspace files, `train.toml` keys, losses, and supported ops are documented in the [di-ml guide](../../README.md).
