#!/usr/bin/env bash
set -ex

if [ ! -f ./sherpa-onnx-supertonic-tts-int8-2026-03-06/duration_predictor.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
  tar xvf sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
  rm sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
fi

dotnet run
