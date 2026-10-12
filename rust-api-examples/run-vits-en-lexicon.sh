#!/usr/bin/env bash
set -ex

if [ ! -d ./vits-piper-en_US-amy-low ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  tar xf vits-piper-en_US-amy-low.tar.bz2
  rm vits-piper-en_US-amy-low.tar.bz2
fi

if [ ! -f ./lexicon-en-us.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
fi

cargo run --example vits_tts_lexicon --   --model ./vits-piper-en_US-amy-low/en_US-amy-low.onnx   --tokens ./vits-piper-en_US-amy-low/tokens.txt   --lexicon ./lexicon-en-us.txt   --debug --output ./generated-vits-en-lexicon-rust.wav   --text "Liliana, the most beautiful and lovely assistant of our team!"
