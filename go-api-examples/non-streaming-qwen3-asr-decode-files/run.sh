#!/usr/bin/env bash

set -ex

export CGO_ENABLED=1

if [ ! -f ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
  tar xvf sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
  rm sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
fi

go mod tidy
go build

./non-streaming-qwen3-asr-decode-files
