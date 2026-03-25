#!/usr/bin/env bash
set -ex

if [ ! -f ./sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/sherpa-onnx-spleeter-2stems-fp16.tar.bz2
  tar xjf sherpa-onnx-spleeter-2stems-fp16.tar.bz2
fi

if [ ! -f ./qi-feng-le-zh.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav
fi

dotnet run
