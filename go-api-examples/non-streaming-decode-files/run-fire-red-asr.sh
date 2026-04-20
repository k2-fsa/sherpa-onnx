#!/usr/bin/env bash

set -ex

export CGO_ENABLED=1

if [ ! -f ./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
  tar xvf sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
  rm sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
  ls -lh sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16
fi

go mod tidy
go build

./non-streaming-decode-files \
  --fire-red-asr-encoder=./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx \
  --fire-red-asr-decoder=./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/decoder.int8.onnx \
  --tokens=./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/tokens.txt \
  ./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/0.wav

