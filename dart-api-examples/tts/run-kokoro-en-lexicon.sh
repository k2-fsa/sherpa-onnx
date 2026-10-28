#!/usr/bin/env bash

set -ex

dart pub get

# please visit
# https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html
# to download more models
if [ ! -f ./kokoro-en-v0_19/model.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
  tar xf kokoro-en-v0_19.tar.bz2
  rm kokoro-en-v0_19.tar.bz2
fi

if [ ! -f ./lexicon-en-us.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
fi

dart run \
  ./bin/kokoro-en-lexicon.dart \
  --model ./kokoro-en-v0_19/model.onnx \
  --voices ./kokoro-en-v0_19/voices.bin \
  --tokens ./kokoro-en-v0_19/tokens.txt \
  --lexicon ./lexicon-en-us.txt \
  --sid 9 \
  --speed 1.0 \
  --output-wav kokoro-en-9-lexicon.wav \
  --text "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."

ls -lh *.wav
