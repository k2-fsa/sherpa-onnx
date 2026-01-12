#!/usr/bin/env bash

set -ex

if [ ! -f ./ten-vad.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
fi

if [ ! -f ./lei-jun-test.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
fi

if [ ! -f ./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
  tar xvf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
  rm sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
fi

dotnet run
