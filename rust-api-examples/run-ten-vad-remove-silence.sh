#!/usr/bin/env bash
set -ex

# https://k2-fsa.github.io/sherpa/onnx/vad/index.html
if [ ! -f "./ten-vad.onnx" ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
fi

if [ ! -f ./lei-jun-test.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
fi

cargo run --example ten_vad_remove_silence -- \
    --input ./lei-jun-test.wav \
    --output ./no-silence-ten-vad.wav \
    --ten-vad-model ./ten-vad.onnx
