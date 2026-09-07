#!/usr/bin/env bash

set -ex

export CGO_ENABLED=1

if [ ! -f ./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
  tar xvf sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
  rm sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
fi

go mod tidy
go build
./non-streaming-medasr-ctc-decode-files
