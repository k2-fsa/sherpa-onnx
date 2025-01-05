#!/usr/bin/env bash

set -ex
if [ ! -f ./vits-piper-en_US-amy-low/en_US-amy-low.onnx ]; then
  curl -OL https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  tar xf vits-piper-en_US-amy-low.tar.bz2
  rm vits-piper-en_US-amy-low.tar.bz2
fi

dotnet run \
  --vits-model=./vits-piper-en_US-amy-low/en_US-amy-low.onnx \
  --tokens=./vits-piper-en_US-amy-low/tokens.txt \
  --data-dir=./vits-piper-en_US-amy-low/espeak-ng-data \
  --debug=1 \
  --output-filename=./amy.wav \
  --text="This is a text to speech application in dotnet with Next Generation Kaldi"

