#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-moonshine-tiny-en-int8/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
  tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
  rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
fi

if [ ! -f ./Obama.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
fi

if [[ ! -f ./silero_vad.onnx ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
fi

dart run \
  ./bin/moonshine.dart \
  --silero-vad ./silero_vad.onnx \
  --preprocessor ./sherpa-onnx-moonshine-tiny-en-int8/preprocess.onnx \
  --encoder ./sherpa-onnx-moonshine-tiny-en-int8/encode.int8.onnx \
  --uncached-decoder ./sherpa-onnx-moonshine-tiny-en-int8/uncached_decode.int8.onnx \
  --cached-decoder ./sherpa-onnx-moonshine-tiny-en-int8/cached_decode.int8.onnx \
  --tokens ./sherpa-onnx-moonshine-tiny-en-int8/tokens.txt \
  --input-wav ./Obama.wav
