#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-streaming-t-one-russian-2025-09-08/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
  tar xvf sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
  rm sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
fi

dart run \
  ./bin/t-one-ctc.dart \
  --model ./sherpa-onnx-streaming-t-one-russian-2025-09-08/model.onnx \
  --tokens ./sherpa-onnx-streaming-t-one-russian-2025-09-08/tokens.txt \
  --input-wav ./sherpa-onnx-streaming-t-one-russian-2025-09-08/0.wav
