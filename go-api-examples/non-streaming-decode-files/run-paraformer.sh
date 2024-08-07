#!/usr/bin/env bash

set -ex

if [ ! -d sherpa-onnx-paraformer-zh-2023-09-14 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
fi

go mod tidy
go build

./non-streaming-decode-files \
  --paraformer ./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx \
  --tokens ./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt \
  --model-type paraformer \
  --debug 0 \
  ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/0.wav
