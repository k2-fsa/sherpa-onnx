#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
  tar xvf sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
  rm sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
fi

dart run \
  ./bin/medasr-ctc.dart \
  --model ./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/model.int8.onnx \
  --tokens ./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/tokens.txt \
  --input-wav ./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/test_wavs/0.wav
