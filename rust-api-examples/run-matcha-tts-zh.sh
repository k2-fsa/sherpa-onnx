#!/usr/bin/env bash
set -ex

if [ ! -f ./matcha-icefall-zh-baker/model-steps-3.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
  tar xvf matcha-icefall-zh-baker.tar.bz2
  rm matcha-icefall-zh-baker.tar.bz2
fi

if [ ! -f ./vocos-22khz-univ.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx
fi

cargo run --example matcha_tts_zh
