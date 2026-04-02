#!/usr/bin/env bash

set -ex

dart pub get

if [[ ! -d ./sherpa-onnx-online-punct-en-2024-08-06 ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
  tar xvf sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
  rm sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
fi

dart run \
  ./bin/online_punctuation.dart \
  --model ./sherpa-onnx-online-punct-en-2024-08-06/model.onnx \
  --bpe-vocab ./sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab
