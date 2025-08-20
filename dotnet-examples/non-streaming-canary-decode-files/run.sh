#!/usr/bin/env bash

set -ex

if [ ! -f sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
  tar xvf sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
  rm sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
fi

dotnet run
