#!/usr/bin/env bash

set -ex

if [ ! -f ./dpdfnet_baseline.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet_baseline.onnx
fi

if [ ! -f ./inp_16k.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav
fi

cargo run --example streaming_speech_enhancement_dpdfnet -- \
  --model ./dpdfnet_baseline.onnx \
  --input ./inp_16k.wav \
  --output ./enhanced-rust-streaming-dpdfnet.wav
