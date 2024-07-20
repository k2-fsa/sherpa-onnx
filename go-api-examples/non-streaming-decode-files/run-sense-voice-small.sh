#!/usr/bin/env bash

set -ex

if [ ! -d sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17  ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
fi

go mod tidy
go build

./non-streaming-decode-files \
  --sense-voice-model ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
  --tokens ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
  --debug 0 \
  ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav
