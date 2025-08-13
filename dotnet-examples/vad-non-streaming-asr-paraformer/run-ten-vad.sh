#!/usr/bin/env bash

set -ex

if [ ! -f ./ten-vad.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
fi

if [ ! -f ./lei-jun-test.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
fi

if [ ! -f ./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

  tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
fi

dotnet run
