#!/usr/bin/env bash

set -ex

if [ ! -f ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
  tar xvf sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
  rm sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2

  ls -lh sherpa-onnx-zipformer-small-audio-tagging-2024-04-15
fi

dotnet run
