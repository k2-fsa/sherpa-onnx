#!/usr/bin/env bash

set -ex


if [ ! -f ./sherpa-onnx-streaming-t-one-russian-2025-09-08/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
  tar xvf sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
  rm sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
fi

go mod tidy
go build

./streaming-decode-files \
  --t-one-ctc ./sherpa-onnx-streaming-t-one-russian-2025-09-08/model.onnx \
  --tokens ./sherpa-onnx-streaming-t-one-russian-2025-09-08/tokens.txt \
  ./sherpa-onnx-streaming-t-one-russian-2025-09-08/0.wav
