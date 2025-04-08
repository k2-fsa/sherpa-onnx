#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/model.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
  tar xvf sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
  rm sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
  ls -lh sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02
fi

dart run \
  ./bin/dolphin-ctc.dart \
  --model ./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/model.int8.onnx \
  --tokens ./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/tokens.txt \
  --input-wav ./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/test_wavs/0.wav
