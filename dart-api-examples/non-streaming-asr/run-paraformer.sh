#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

  tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
fi

dart run \
  ./bin/paraformer.dart \
  --model ./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx \
  --tokens ./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt \
  --input-wav ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/3-sichuan.wav
