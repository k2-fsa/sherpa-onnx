#!/usr/bin/env bash

set -ex

dart pub get

if [[ ! -f ./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
  tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
  rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
fi

dart run \
  ./bin/punctuations.dart \
  --model ./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx
