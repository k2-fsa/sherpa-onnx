#!/usr/bin/env bash

set -ex

export CGO_ENABLED=1

if [ ! -f ./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2
  tar xvf sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2
  rm sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2
fi

go mod tidy
go build

./non-streaming-w2v-bert-ctc-decode-files
