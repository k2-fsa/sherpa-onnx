#!/usr/bin/env bash

set -ex

if [ ! -d vits-ljs ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-ljs.tar.bz2
  tar xvf vits-ljs.tar.bz2
  rm vits-ljs.tar.bz2
fi

go mod tidy
go build

./non-streaming-tts \
  --vits-model=./vits-ljs/vits-ljs.onnx \
  --vits-lexicon=./vits-ljs/lexicon.txt \
  --vits-tokens=./vits-ljs/tokens.txt \
  --sid=0 \
  --debug=1 \
  --output-filename=./vits-ljs.wav \
  "Liliana, the most beautiful and lovely assistant of our team!"
