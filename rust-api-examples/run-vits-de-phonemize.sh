#!/usr/bin/env bash
set -ex

if [ ! -d ./vits-piper-de_DE-glados-high ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-de_DE-glados-high.tar.bz2
  tar xf vits-piper-de_DE-glados-high.tar.bz2
  rm vits-piper-de_DE-glados-high.tar.bz2
fi

cargo run --features phonemize --example vits_tts_de_phonemize
