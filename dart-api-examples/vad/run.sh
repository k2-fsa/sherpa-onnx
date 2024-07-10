#!/usr/bin/env bash

set -ex

dart pub get


if [[ ! -f ./silero_vad.onnx ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
fi

if [[ ! -f ./lei-jun-test.wav ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
fi

dart run \
  ./bin/vad.dart \
  --silero-vad ./silero_vad.onnx \
  --input-wav ./lei-jun-test.wav \
  --output-wav ./lei-jun-test-no-silence.wav

ls -lh *.wav
