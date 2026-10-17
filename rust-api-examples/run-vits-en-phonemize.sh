#!/usr/bin/env bash
set -ex

if [ ! -d ./vits-piper-en_US-amy-low ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  tar xf vits-piper-en_US-amy-low.tar.bz2
  rm vits-piper-en_US-amy-low.tar.bz2
fi

cargo run --features phonemize --example vits_tts_phonemize
