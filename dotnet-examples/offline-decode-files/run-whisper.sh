#!/usr/bin/env bash

set -ex

git lfs install

git clone https://huggingface.co/csukuangfj/sherpa-onnx-whisper-large-v3
ls -lh sherpa-onnx-whisper-large-v3

dotnet run \
  --num-threads=2 \
  --whisper-encoder=./sherpa-onnx-whisper-large-v3/large-v3-encoder.int8.onnx \
  --whisper-decoder=./sherpa-onnx-whisper-large-v3/large-v3-decoder.int8.onnx \
  --tokens=./sherpa-onnx-whisper-large-v3/large-v3-tokens.txt \
  --files ./sherpa-onnx-whisper-large-v3/test_wavs/0.wav \
  ./sherpa-onnx-whisper-large-v3/test_wavs/1.wav \
  ./sherpa-onnx-whisper-large-v3/test_wavs/8k.wav

dotnet run \
  --num-threads=2 \
  --whisper-encoder=./sherpa-onnx-whisper-large-v3/large-v3-encoder.onnx \
  --whisper-decoder=./sherpa-onnx-whisper-large-v3/large-v3-decoder.onnx \
  --tokens=./sherpa-onnx-whisper-large-v3/large-v3-tokens.txt \
  --files ./sherpa-onnx-whisper-large-v3/test_wavs/0.wav \
  ./sherpa-onnx-whisper-large-v3/test_wavs/1.wav \
  ./sherpa-onnx-whisper-large-v3/test_wavs/8k.wav
