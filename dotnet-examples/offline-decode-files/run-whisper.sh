#!/usr/bin/env bash

set -ex

if [ ! -d ./sherpa-onnx-whisper-tiny.en ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en
  cd sherpa-onnx-whisper-tiny.en
  git lfs pull --include "*.onnx"
  cd ..
fi

dotnet run \
  --num-threads=2 \
  --whisper-encoder=./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx \
  --whisper-decoder=./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx \
  --tokens=./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt \
  --files ./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav \
  ./sherpa-onnx-whisper-tiny.en/test_wavs/1.wav \
  ./sherpa-onnx-whisper-tiny.en/test_wavs/8k.wav
