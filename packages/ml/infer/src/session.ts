import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-web';

export type TensorData = Float32Array | Int32Array | BigInt64Array;

export type Tensor<Data extends TensorData = Float32Array> = {
  data: Data;
  dims: number[];
};

export type SessionOptions = {
  /** Default `["wasm"]`. In a WebGPU browser pass `["webgpu", "wasm"]`. */
  executionProviders?: string[];
};

let wasmConfigured = false;

function configureWasm(): void {
  if (wasmConfigured) {
    return;
  }
  wasmConfigured = true;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  try {
    const pkg = fileURLToPath(import.meta.resolve('onnxruntime-web/package.json'));
    ort.env.wasm.wasmPaths = `${join(dirname(pkg), 'dist')}/`;
  } catch {
    // Bundlers that inlined the package still serve wasm from their own public path.
  }
}

export class Session {
  private released = false;
  private constructor(
    private readonly inner: ort.InferenceSession,
    readonly inputs: string[],
    readonly outputs: string[],
  ) {}

  static async fromBytes(bytes: Uint8Array, options: SessionOptions = {}): Promise<Session> {
    configureWasm();
    const providers = options.executionProviders ?? ['wasm'];
    const inner = await ort.InferenceSession.create(bytes, {
      executionProviders: providers,
    });
    return new Session(inner, [...inner.inputNames], [...inner.outputNames]);
  }

  static async fromPath(path: string, options: SessionOptions = {}): Promise<Session> {
    const bytes = new Uint8Array(await readFile(path));
    return Session.fromBytes(bytes, options);
  }

  async run(
    feeds: Record<string, Tensor<TensorData>>,
  ): Promise<Record<string, Tensor<TensorData>>> {
    if (this.released) throw new Error('Session has been released');
    const ortFeeds: Record<string, ort.Tensor> = {};
    for (const [name, tensor] of Object.entries(feeds)) {
      const type =
        tensor.data instanceof Float32Array
          ? 'float32'
          : tensor.data instanceof Int32Array
            ? 'int32'
            : tensor.data instanceof BigInt64Array
              ? 'int64'
              : undefined;
      if (!type) throw new Error(`unsupported input tensor type: ${name}`);
      ortFeeds[name] = new ort.Tensor(type, tensor.data, tensor.dims);
    }
    const results = await this.inner.run(ortFeeds);
    const out: Record<string, Tensor<TensorData>> = {};
    for (const name of this.outputs) {
      const t = results[name];
      if (!t) {
        throw new Error(`missing output ${name}`);
      }
      if (
        !(
          t.data instanceof Float32Array ||
          t.data instanceof Int32Array ||
          t.data instanceof BigInt64Array
        )
      ) {
        throw new Error(`unsupported output tensor type: ${name} (${t.type})`);
      }
      out[name] = { data: t.data, dims: [...t.dims] };
    }
    return out;
  }

  /** Release runtime memory after the final run. Safe to call more than once. */
  async release(): Promise<void> {
    if (this.released) return;
    this.released = true;
    await this.inner.release();
  }
}
