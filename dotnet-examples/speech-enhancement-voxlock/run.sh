#!/usr/bin/env bash
set -ex

if [ ! -d ./sherpa-onnx-voxlock-int8 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2
  tar xvf sherpa-onnx-voxlock-int8.tar.bz2
  rm sherpa-onnx-voxlock-int8.tar.bz2
fi

dotnet run
