#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2

  tar xvf sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
  rm sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
fi

if [ ! -f ./lei-jun-test.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
fi

if [[ ! -f ./silero_vad.onnx ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
fi

dart run \
  ./bin/zipformer-ctc.dart \
  --silero-vad ./silero_vad.onnx \
  --model ./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/model.int8.onnx \
  --tokens ./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/tokens.txt \
  --input-wav ./lei-jun-test.wav
