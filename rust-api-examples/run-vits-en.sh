#!/usr/bin/env bash
set -ex

if [ ! -d ./vits-piper-en_US-amy-low ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  tar xf vits-piper-en_US-amy-low.tar.bz2
  rm vits-piper-en_US-amy-low.tar.bz2
fi

cargo run --example vits_tts --   --model ./vits-piper-en_US-amy-low/en_US-amy-low.onnx   --tokens ./vits-piper-en_US-amy-low/tokens.txt   --data-dir ./vits-piper-en_US-amy-low/espeak-ng-data   --output ./generated-vits-en-rust.wav   --text "Liliana, the most beautiful and lovely assistant of our team!"
