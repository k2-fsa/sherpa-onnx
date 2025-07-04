#!/usr/bin/env bash

set -ex

if [ ! -f ./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2

  tar xvf sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
  rm sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
fi

go mod tidy
go build

./non-streaming-decode-files \
  --zipformer-ctc ./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/model.int8.onnx \
  --tokens ./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/tokens.txt \
  --debug 0 \
  ./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/test_wavs/0.wav
