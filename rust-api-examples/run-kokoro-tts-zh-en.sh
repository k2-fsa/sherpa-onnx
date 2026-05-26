#!/usr/bin/env bash
set -ex

if [ ! -f ./kokoro-multi-lang-v1_0/model.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
  tar xf kokoro-multi-lang-v1_0.tar.bz2
  rm kokoro-multi-lang-v1_0.tar.bz2
fi

cargo run --example kokoro_tts_zh_en
