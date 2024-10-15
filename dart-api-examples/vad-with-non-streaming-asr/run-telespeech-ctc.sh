#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2

  tar xvf sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2
  rm sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2
fi

if [ ! -f ./lei-jun-test.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
fi

if [[ ! -f ./silero_vad.onnx ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
fi

dart run \
  ./bin/telespeech-ctc.dart \
  --silero-vad ./silero_vad.onnx \
  --model ./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/model.int8.onnx \
  --tokens ./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/tokens.txt \
  --input-wav ./lei-jun-test.wav
