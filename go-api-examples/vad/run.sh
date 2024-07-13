#!/usr/bin/env bash

set -ex

if [ ! -f ./silero_vad.onnx ]; then
  curl -SL -O https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
fi

go mod tidy
go build
./vad
