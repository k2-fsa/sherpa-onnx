#!/usr/bin/env bash

set -ex

if [ ! -f ./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/encoder.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2
  tar xvf sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2
  rm sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2
fi

dotnet run
