#!/usr/bin/env bash
set -ex

if [ ! -f ./matcha-icefall-en_US-ljspeech/model-steps-3.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
  tar xf matcha-icefall-en_US-ljspeech.tar.bz2
  rm matcha-icefall-en_US-ljspeech.tar.bz2
fi

if [ ! -f ./vocos-22khz-univ.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx
fi

cargo run --example matcha_tts_en
