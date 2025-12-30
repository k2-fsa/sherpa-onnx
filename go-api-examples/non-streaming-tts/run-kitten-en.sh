#!/usr/bin/env bash

set -ex

if [ ! -f ./kitten-nano-en-v0_1-fp16/model.fp16.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_1-fp16.tar.bz2
  tar xf kitten-nano-en-v0_1-fp16.tar.bz2
  rm kitten-nano-en-v0_1-fp16.tar.bz2
fi

go mod tidy
go build

./non-streaming-tts \
  --kitten-model=./kitten-nano-en-v0_1-fp16/model.fp16.onnx \
  --kitten-voices=./kitten-nano-en-v0_1-fp16/voices.bin \
  --kitten-tokens=./kitten-nano-en-v0_1-fp16/tokens.txt \
  --kitten-data-dir=./kitten-nano-en-v0_1-fp16/espeak-ng-data \
  --debug=1 \
  --output-filename=./test-kitten-en.wav \
  "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
