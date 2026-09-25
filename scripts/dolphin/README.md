# Dolphin attention decoder export

This directory contains scripts for exporting the [Dolphin](https://github.com/DataoceanAI/Dolphin)
encoder and attention decoder to ONNX, enabling language/region control in
sherpa-onnx (the released `model.onnx` contains only the CTC branch, which
ignores the prompt prefix).

## Usage

```bash
# 1. Clone the reference implementation
git clone https://github.com/DataoceanAI/Dolphin /path/to/dolphin

# 2. Download model files (base.pt, train.yaml, units.txt, bpe.model,
#    feats_stats.npz) from https://huggingface.co/DataoceanAI/dolphin-base
#    into /path/to/dolphin-base

# 3. Export
python3 ./export-onnx.py \
    --dolphin-repo /path/to/dolphin \
    --model-dir /path/to/dolphin-base \
    --output-dir ./out

# 4. Verify against the PyTorch reference (greedy decode parity)
python3 ./test-onnx.py \
    --dolphin-repo /path/to/dolphin \
    --model-dir /path/to/dolphin-base \
    --onnx-dir ./out \
    --wav /path/to/some.wav \
    --sherpa-onnx-offline /path/to/build/bin/sherpa-onnx-offline

# Optional: force language/region
python3 ./test-onnx.py ... --lang zh --region CN
```

## Exported files

The export also writes tokens.txt from the model's units.txt. Use the
encoder, decoder and tokens from the same export.

- `encoder.onnx`: `feats [1,T,80]` + `feats_len [1]` -> `encoder_out [1,T',512]`.
  Global CMVN is stripped from the graph; `mean`/`invstd` are stored as model
  metadata (comma-separated), the same convention as the released CTC
  `model.onnx`, and sherpa-onnx normalizes features in C++.
- `decoder.onnx`: `encoder_out [1,T',512]` + `ys [1,N]` (int64 token prefix
  starting with `<sos>`) -> `logp [1,vocab_size]` for the next token. It
  re-encodes the whole prefix every step (no KV cache).

## sherpa-onnx side

```bash
sherpa-onnx-offline \
  --dolphin-encoder=encoder.onnx \
  --dolphin-decoder=decoder.onnx \
  --dolphin-language=zh \
  --dolphin-region=CN \
  --tokens=tokens.txt \
  input.wav
```

`--dolphin-language`/`--dolphin-region` are optional; when omitted the decoder
predicts them. `--dolphin-region` requires `--dolphin-language`. If only
`--dolphin-model` is given, the original CTC path is used unchanged.

Unsupported language/region tokens at construction produce a warning and
fall back to automatic detection. An invalid SetConfig update is ignored,
preserving the previous valid prompt. SetConfig changes language/region only;
it does not reload model files.

## Build and packaging compatibility

Build the native library and language bindings from the same source revision.
The Dolphin C configuration has grown, changing the layout of its enclosing
offline configurations; mixing a previously compiled client with the new
native library (or the reverse) is not supported.

Package.swift still pins a released CTC-only XCFramework. Swift Package builds
keep that release's CTC interface; attention decoding requires a source-built
XCFramework and the matching non-package Swift wrapper. The package dependency
and wrapper must be updated together when an attention-capable binary is released.

## Regression tests without model downloads

Build the official CMake targets sherpa-onnx-offline and
offline-recognizer-dolphin-impl-test with SHERPA_ONNX_ENABLE_TESTS=ON.
Set SHERPA_ONNX_OFFLINE and SHERPA_ONNX_DOLPHIN_TEST to those executables, then run
python scripts/dolphin/test_attention.py -v from the repository root.
The tests generate tiny ONNX fixtures in a temporary directory, exercise the
native recognizer and configuration updates, and require no trained weights.
They test wiring, not recognition accuracy.

The export verifier compares encoder tensors with configurable --rtol/--atol,
the entire PyTorch/ONNX token sequence (including header and EOS), and native
recognizer tokens and language. A mismatch or failed native process exits nonzero.
