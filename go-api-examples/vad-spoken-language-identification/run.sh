#!/usr/bin/env bash

set -ex

if [ ! -f ./silero_vad.onnx ]; then
  curl -SL -O https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
fi

if [ ! -f ./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
  tar xvf sherpa-onnx-whisper-tiny.tar.bz2
  rm sherpa-onnx-whisper-tiny.tar.bz2
fi

go mod tidy
go build
./vad-spoken-language-identification
