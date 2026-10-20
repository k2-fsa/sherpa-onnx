#!/usr/bin/env bash

set -ex

export CGO_ENABLED=1

if [ ! -f ./matcha-icefall-en_US-ljspeech/model-steps-3.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
  tar xf matcha-icefall-en_US-ljspeech.tar.bz2
  rm matcha-icefall-en_US-ljspeech.tar.bz2
fi

if [ ! -f ./vocos-22khz-univ.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx
fi

go mod tidy
go build

./non-streaming-tts-phonemize \
  --matcha-acoustic-model=./matcha-icefall-en_US-ljspeech/model-steps-3.onnx \
  --matcha-vocoder=./vocos-22khz-univ.onnx \
  --matcha-tokens=./matcha-icefall-en_US-ljspeech/tokens.txt \
  --debug=1 \
  --output-filename=./test-matcha-en-phonemize.wav \
  "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
