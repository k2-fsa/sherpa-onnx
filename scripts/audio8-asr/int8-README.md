# sherpa-onnx-audio8-asr-0.1b-int8 (dynamic int8)

Audio8-ASR-0.1B converted to ONNX for sherpa-onnx.

- Upstream model: https://huggingface.co/Edge0/Audio8-ASR-0.1B
- Exported by: https://github.com/k2-fsa/sherpa-onnx (scripts/audio8-asr)
- License: Creative Commons Attribution Non Commercial 4.0 (see `LICENSE`)
  — the checkpoint is **non-commercial**

## What is in this directory

| file | description |
|---|---|
| `encoder.int8.onnx` | Whisper-style log-mel `[1, 128, 3000]` (+ `num_rows`) → audio hidden states `[1, 390, 1024]` |
| `decoder.int8.onnx` | audio projector + 8-layer Qwen2 decoder with a static 512-slot KV cache |
| `tokenizer/` | `vocab.json`, `merges.txt`, `tokenizer_config.json`, `added_tokens.json`, `tokenizer.json` |
| `test_wavs/` | one 16 kHz test clip per supported language |

The model is an autoregressive ASR system: an audio encoder (18 self-attention
layers, hidden 896) followed by a 0.1B Qwen2 decoder. It transcribes Chinese,
English, French, German, Japanese, Korean and Cantonese, in 30 s windows.

## Usage

```bash
./build/bin/sherpa-onnx-offline \
  --audio8-asr-encoder=./sherpa-onnx-audio8-asr-0.1b-int8/encoder.int8.onnx \
  --audio8-asr-decoder=./sherpa-onnx-audio8-asr-0.1b-int8/decoder.int8.onnx \
  --audio8-asr-tokenizer=./sherpa-onnx-audio8-asr-0.1b-int8/tokenizer \
  --num-threads=4 \
  ./sherpa-onnx-audio8-asr-0.1b-int8/test_wavs/en.wav
```

```python
import sherpa_onnx

recognizer = sherpa_onnx.OfflineRecognizer.from_audio8_asr(
    encoder="./sherpa-onnx-audio8-asr-0.1b-int8/encoder.onnx",
    decoder="./sherpa-onnx-audio8-asr-0.1b-int8/decoder.onnx",
    tokenizer="./sherpa-onnx-audio8-asr-0.1b-int8/tokenizer",
    num_threads=4,
)
```

See https://k2-fsa.github.io/sherpa/onnx/ for the full documentation.
