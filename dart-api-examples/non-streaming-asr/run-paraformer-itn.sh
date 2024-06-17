#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2

  tar xvf sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2
  rm sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2
fi

if [ ! -f ./itn-zh-number.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav
fi

if [ ! -f ./itn_zh_number.fst ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
fi

dart run \
  ./bin/paraformer-itn.dart \
  --model ./sherpa-onnx-paraformer-zh-2023-03-28/model.int8.onnx \
  --tokens ./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
  --rule-fsts ./itn_zh_number.fst \
  --input-wav ./itn-zh-number.wav
