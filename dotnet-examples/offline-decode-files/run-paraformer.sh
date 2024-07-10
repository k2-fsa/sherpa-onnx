#!/usr/bin/env bash

set -ex

if [ ! -d ./sherpa-onnx-paraformer-zh-2023-09-14 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
fi

dotnet run \
  --tokens=./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt \
  --paraformer=./sherpa-onnx-paraformer-zh-2023-09-14/model.onnx \
  --num-threads=2 \
  --files ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/0.wav \
  ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/1.wav \
  ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/2.wav \
  ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/8k.wav
