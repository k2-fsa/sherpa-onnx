#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./dpdfnet_baseline.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet_baseline.onnx
fi

if [ ! -f ./inp_16k.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav
fi

dart run \
  ./bin/speech_enhancement_dpdfnet.dart \
  --model ./dpdfnet_baseline.onnx \
  --input-wav ./inp_16k.wav \
  --output-wav ./enhanced-16k.wav

ls -lh *.wav
