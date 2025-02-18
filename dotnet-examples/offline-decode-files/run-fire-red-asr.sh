#!/usr/bin/env bash

set -ex

if [ ! -f ./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
  tar xvf sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
  rm sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
  ls -lh sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16
fi

dotnet run \
  --num-threads=2 \
  --fire-red-asr-encoder=./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx \
  --fire-red-asr-decoder=./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/decoder.int8.onnx \
  --tokens=./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/tokens.txt \
  --files ./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/0.wav
