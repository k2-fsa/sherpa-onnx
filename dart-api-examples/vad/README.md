# Introduction

This example shows how to use the Dart API from sherpa-onnx for voice activity detection (VAD).
Specifically, we use VAD to remove silences from a wave file.

# Usage

```bash
dart pub get

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav

dart run \
  ./bin/vad.dart \
  --silero-vad ./silero_vad.onnx \
  --input-wav ./lei-jun-test.wav \
  --output-wav ./lei-jun-test-no-silence.wav
```

It should generate a file `lei-jun-test-no-silence.wav`, where silences are removed.
