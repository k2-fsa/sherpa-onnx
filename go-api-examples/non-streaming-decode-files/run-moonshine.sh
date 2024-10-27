#!/usr/bin/env bash

set -ex

if [ ! -f ./sherpa-onnx-moonshine-tiny-en-int8/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
  tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
  rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
fi

go mod tidy
go build

./non-streaming-decode-files \
  --moonshine-preprocessor=./sherpa-onnx-moonshine-tiny-en-int8/preprocess.onnx \
  --moonshine-encoder=./sherpa-onnx-moonshine-tiny-en-int8/encode.int8.onnx \
  --moonshine-uncached-decoder=./sherpa-onnx-moonshine-tiny-en-int8/uncached_decode.int8.onnx \
  --moonshine-cached-decoder=./sherpa-onnx-moonshine-tiny-en-int8/cached_decode.int8.onnx \
  --tokens=./sherpa-onnx-moonshine-tiny-en-int8/tokens.txt \
  ./sherpa-onnx-moonshine-tiny-en-int8/test_wavs/0.wav

