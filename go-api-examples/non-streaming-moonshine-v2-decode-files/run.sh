#!/usr/bin/env bash

set -ex

export CGO_ENABLED=1

if [ ! -f ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/encoder_model.ort ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
  tar xvf sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
  rm sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
fi

go mod tidy
go build
./non-streaming-moonshine-v2-decode-files
