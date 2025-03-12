#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./gtcrn_simple.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
fi

if [ ! -f ./inp_16k.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav
fi


dart run \
  ./bin/speech_enhancement_gtcrn.dart \
  --model ./gtcrn_simple.onnx \
  --input-wav ./inp_16k.wav \
  --output-wav ./enhanced-16k.wav

ls -lh *.wav
