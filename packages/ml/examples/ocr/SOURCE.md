# OCR fixture

First real (non-XOR) model in this repo: **English text-line recognition**, not a detector.

| Field | Value |
| --- | --- |
| File | `models/en_PP-OCRv4_rec_mobile.onnx` (fetched, not committed) |
| Source | RapidOCR ONNX conversion of PaddleOCR `en_PP-OCRv4_mobile_rec` |
| URL | https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.onnx |
| sha256 | `e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318` |
| Size | 7.3 MiB |
| License | Apache-2.0 (PaddleOCR / RapidOCR) |
| Dict | `en_dict.txt` (generated: ASCII 48–126, then 33–47, then space; CTC blank is index 0) |
| Input | `x` float32 `[N, 3, H, W]` (BGR, height 48, padded width 320, values in `[-1, 1]`) |
| Output | `softmax_2.tmp_0` float32 `[N, T, 97]` |

```bash
bun scripts/fetch-ocr-model.ts
```

This command and the OCR test generate the dictionary automatically, including
when the model is already cached. An existing dictionary with a different
character order is rejected. Neither the dictionary nor model is committed.

`@di-framework/ml` runs this graph via ONNX Runtime Web. `di-ml` **does not train it**: Conv and the rest of the vision subset are lint-only (P3). `ExecGraph::load` must reject the file with a missing-op list.

Ops in this graph (from the ONNX): Add, AveragePool, BatchNormalization, Cast, Concat, Conv, Div, GlobalAveragePool, HardSigmoid, HardSwish, MatMul, Mul, Pad, ReduceMean, Relu, Reshape, Shape, Sigmoid, Slice, Softmax, Squeeze, Transpose.

`di-ml` Ready subset covers the elementwise / reshape / MatMul / Softmax slice. **Not trained:** Conv, BatchNormalization, HardSwish, HardSigmoid, AveragePool, GlobalAveragePool, Pad (P2). Lint tags those as P3 (Pad P2).

The line image is rendered in the test (`renderGlyphLine("HELLO")`), not a scanned page. Detection + layout are out of scope for this fixture.
