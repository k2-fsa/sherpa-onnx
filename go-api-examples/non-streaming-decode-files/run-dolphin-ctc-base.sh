#!/usr/bin/env bash

set -ex

if [ ! -f ./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/model.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
  tar xvf sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
  rm sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
  ls -lh sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02
fi

go mod tidy
go build

./non-streaming-decode-files \
  --dolphin-model ./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/model.int8.onnx \
  --tokens ./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/tokens.txt \
  --debug 0 \
  ./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/test_wavs/0.wav
