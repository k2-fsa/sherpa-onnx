#!/usr/bin/env bash

set -ex

export CGO_ENABLED=1

if [ ! -f ./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/duration_predictor.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
  tar xvf sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
  rm sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
fi

go mod tidy
go build

./supertonic-tts
