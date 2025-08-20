#!/usr/bin/env bash

set -ex

dart pub get

# please visit
# https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kitten.html
# to download more models
if [ ! -f ./kitten-nano-en-v0_1-fp16/model.fp16.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_1-fp16.tar.bz2
  tar xf kitten-nano-en-v0_1-fp16.tar.bz2
  rm kitten-nano-en-v0_1-fp16.tar.bz2
fi

dart run \
  ./bin/kitten-en.dart \
  --model ./kitten-nano-en-v0_1-fp16/model.fp16.onnx \
  --voices ./kitten-nano-en-v0_1-fp16/voices.bin \
  --tokens ./kitten-nano-en-v0_1-fp16/tokens.txt \
  --data-dir ./kitten-nano-en-v0_1-fp16/espeak-ng-data \
  --sid 0 \
  --speed 1.0 \
  --output-wav kitten-en-0.wav \
  --text "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."

ls -lh *.wav
