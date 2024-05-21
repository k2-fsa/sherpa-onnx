#!/usr/bin/env bash

set -ex

if [ ! -d vits-piper-en_US-lessac-medium ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-lessac-medium.tar.bz2
  tar xvf vits-piper-en_US-lessac-medium.tar.bz2
  rm vits-piper-en_US-lessac-medium.tar.bz2
fi

go mod tidy
go build

./non-streaming-tts \
  --vits-model=./vits-piper-en_US-lessac-medium/en_US-lessac-medium.onnx \
  --vits-data-dir=./vits-piper-en_US-lessac-medium/espeak-ng-data \
  --vits-tokens=./vits-piper-en_US-lessac-medium/tokens.txt \
  --output-filename=./liliana-piper-en_US-lessac-medium.wav \
  'liliana, the most beautiful and lovely assistant of our team!'
