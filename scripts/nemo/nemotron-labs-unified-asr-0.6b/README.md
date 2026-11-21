# Nemotron Labs Unified ASR 0.6B

Export `nvidia/nemotron-labs-unified-asr-0.6b` for sherpa-onnx's buffered
Parakeet Unified runtime. The encoder includes the language prompt layer and
accepts `prompt_index` as its third input, like the cache-aware NeMo models.
Language IDs come from ONNX metadata. No extra runtime model file is needed.

## Export

Install NeMo with multilingual unified support and ONNX export dependencies:

```bash
pip install "nemo_toolkit[asr] @ https://github.com/NVIDIA/NeMo/archive/43dd97ddf.tar.gz"
pip install onnx onnxruntime ipython
python3 export_onnx.py --left-context 70
```

The default downloads the public checkpoint at revision `023ce907`.
Use `--model /path/to/model.nemo` for a local checkpoint. Run the script in a
separate output directory for each context setting.

| `--left-context` | Left context | Chunk | Right look-ahead |
|---|---|---|---|
| 70 (default) | 5.60 s | 0.56 s | 0.56 s |
| 37 | 2.96 s | 0.56 s | 0.56 s |

The attention context is baked into each export. Both settings have 1.12 s
chunk-plus-look-ahead latency. They differ in encoder work and accuracy.

The runtime files are `encoder.int8.onnx` (or `encoder.onnx` and its external
weights), `decoder.onnx`, `joiner.onnx`, and `tokens.txt`.
The int8 recipe quantizes only the acoustic encoder's MatMul weights with
per-channel QUInt8. The language prompt, decoder, and joiner remain fp32.
Intermediate `encoder.raw*` and `prompt.onnx` files are export inputs and are
not needed by the runtime.

`./run.sh 70` (or `37`) installs the export dependencies, tests composition,
exports the models, and runs the CLI with `en`, `en-US`, and `auto`. Build the
streaming CLI in the repository's `build/` directory first, or set
`SHERPA_ONNX_BIN` to its path. Set `NEMO_MODEL` to reuse a local checkpoint.
The script creates
`sherpa-onnx-nemotron-labs-unified-asr-0.6b-int8-streaming-1120ms-L70.tar.bz2`
(or `L37`), including the pinned model license, model card, supporting
documentation, and a sample WAV. The export workflow publishes these packages
using the same Hugging Face and `asr-models` release steps as the English model.

## Decode

```bash
sherpa-onnx \
  --encoder=encoder.int8.onnx \
  --decoder=decoder.onnx \
  --joiner=joiner.onnx \
  --tokens=tokens.txt \
  --language=fr \
  test.wav
```

Python callers set the existing per-stream option:

```python
stream = recognizer.create_stream()
stream.set_option("language", "fr")
```

C, C++, Flutter, and Swift callers use their existing stream option setter
with the same `language` key. Empty, `auto`, and unsupported language strings
use the model's auto prompt. Locale spellings such as `en-US` and `en_US`
resolve to the corresponding base language when metadata uses bare codes.
A reset clears segment results while keeping buffered encoder and predictor
history. Create a new stream to start with empty context.
