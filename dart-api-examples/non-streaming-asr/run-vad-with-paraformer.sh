#!/usr/bin/env bash

set -ex

dart pub get

if [[ ! -f ./silero_vad.onnx ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
fi

if [[ ! -f ./lei-jun-test.wav ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
fi

if [ ! -f ./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

  tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
fi

dart run \
  ./bin/vad-with-paraformer.dart \
  --silero-vad ./silero_vad.onnx \
  --model ./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx \
  --tokens ./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt \
  --input-wav ./lei-jun-test.wav
