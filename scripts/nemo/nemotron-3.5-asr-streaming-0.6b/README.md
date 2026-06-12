# Multilingual Nemotron-3.5 Streaming ASR

This directory exports the NVIDIA NeMo model
`nvidia/nemotron-3.5-asr-streaming-0.6b` to sherpa-onnx streaming transducer
packages.

The exporter writes the same package layout as
`nvidia/nemotron-speech-streaming-en-0.6b`:

- `encoder.onnx`
- `encoder.data`
- `decoder.onnx`
- `joiner.onnx`
- int8 variants for all three ONNX graphs
- `tokens.txt` converted from the model's SentencePiece tokenizer

The encoder metadata contains `prompt_dictionary` and `auto_prompt_id`. Users
set the per-stream language as a string; the numerical prompt id is internal.
An empty language string and `auto` use the model's auto-detect prompt.

## Export

As of June 2026, stable NeMo releases lack `EncDecRNNTBPEModelWithPrompt`,
so install NeMo from git main.

```bash
pip install Cython packaging
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"
pip install onnxruntime ipython sentencepiece
pip install kaldi-native-fbank
pip install soundfile librosa

python3 ./export_onnx.py
```

The script exports 80ms, 160ms, 560ms, and 1120ms chunk sizes.

## Decode

Forced language:

```bash
./build/bin/sherpa-onnx \
  --encoder=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/encoder.int8.onnx \
  --decoder=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/decoder.int8.onnx \
  --joiner=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/joiner.int8.onnx \
  --tokens=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/tokens.txt \
  --language=ja \
  ./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/test_wavs/ja.wav
```

Auto language:

```bash
./build/bin/sherpa-onnx \
  --encoder=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/encoder.int8.onnx \
  --decoder=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/decoder.int8.onnx \
  --joiner=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/joiner.int8.onnx \
  --tokens=./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/tokens.txt \
  --language=auto \
  ./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/test_wavs/ja.wav
```

The same auto behavior is used when `--language` is omitted.
