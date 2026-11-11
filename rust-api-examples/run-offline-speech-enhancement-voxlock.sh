#!/usr/bin/env bash

set -ex

if [ ! -d ./sherpa-onnx-voxlock-int8 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2
  tar xvf sherpa-onnx-voxlock-int8.tar.bz2
  rm sherpa-onnx-voxlock-int8.tar.bz2
fi

cargo run --example offline_speech_enhancement_voxlock -- \
  --enroll-model ./sherpa-onnx-voxlock-int8/enroll.onnx \
  --chunk-model ./sherpa-onnx-voxlock-int8/chunk_int8.onnx \
  --input ./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav \
  --output ./voxlock-enhanced-16k.wav

mkdir -p offline_speech_enhancement_voxlock
mv voxlock-enhanced-16k.wav offline_speech_enhancement_voxlock/
cp sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav offline_speech_enhancement_voxlock/
