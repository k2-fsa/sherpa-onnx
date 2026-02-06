#!/usr/bin/env bash

set -ex

export CGO_ENABLED=1

if [ ! -f ./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
  tar xvf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
  rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
fi

go mod tidy
go build

./zero-shot-pocket-tts \
  --reference-audio ./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav \
  --output-filename ./generated-bria.wav \
  --text "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar."
