#!/usr/bin/env bash

set -ex

if [ ! -f ./kokoro-en-v0_19/model.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
  tar xf kokoro-en-v0_19.tar.bz2
  rm kokoro-en-v0_19.tar.bz2
fi

go mod tidy
go build

./non-streaming-tts \
  --kokoro-model=./kokoro-en-v0_19/model.onnx \
  --kokoro-voices=./kokoro-en-v0_19/voices.bin \
  --kokoro-tokens=./kokoro-en-v0_19/tokens.txt \
  --kokoro-data-dir=./kokoro-en-v0_19/espeak-ng-data \
  --debug=1 \
  --output-filename=./test-kokoro-en.wav \
  "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
