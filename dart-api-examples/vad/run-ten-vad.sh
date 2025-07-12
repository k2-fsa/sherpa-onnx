#!/usr/bin/env bash

set -ex

dart pub get


if [[ ! -f ./ten-vad.onnx ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
fi

if [[ ! -f ./lei-jun-test.wav ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
fi

dart run \
  ./bin/ten-vad.dart \
  --ten-vad ./ten-vad.onnx \
  --input-wav ./lei-jun-test.wav \
  --output-wav ./lei-jun-test-no-silence.wav

ls -lh *.wav
