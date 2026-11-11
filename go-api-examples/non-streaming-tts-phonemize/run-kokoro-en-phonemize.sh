#!/usr/bin/env bash

set -ex

export CGO_ENABLED=1

if [ ! -f ./kokoro-en-v0_19/model.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
  tar xf kokoro-en-v0_19.tar.bz2
  rm kokoro-en-v0_19.tar.bz2
fi

go mod tidy
go build

./non-streaming-tts-phonemize \
  --kokoro-model=./kokoro-en-v0_19/model.onnx \
  --kokoro-voices=./kokoro-en-v0_19/voices.bin \
  --kokoro-tokens=./kokoro-en-v0_19/tokens.txt \
  --debug=1 \
  --output-filename=./test-kokoro-en-phonemize.wav \
  "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
