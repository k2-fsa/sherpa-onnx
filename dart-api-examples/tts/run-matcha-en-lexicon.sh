#!/usr/bin/env bash

set -ex

dart pub get

# please visit
# https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html#matcha-icefall-en-us-ljspeech-american-english-1-female-speaker
# matcha.html#matcha-icefall-en-us-ljspeech-american-english-1-female-speaker
# to download more models
if [ ! -f ./matcha-icefall-en_US-ljspeech/model-steps-3.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
  tar xf matcha-icefall-en_US-ljspeech.tar.bz2
  rm matcha-icefall-en_US-ljspeech.tar.bz2
fi

if [ ! -f ./vocos-22khz-univ.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx
fi

if [ ! -f ./lexicon-en-us.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
fi

dart run \
  ./bin/matcha-en-lexicon.dart \
  --acoustic-model ./matcha-icefall-en_US-ljspeech/model-steps-3.onnx \
  --vocoder ./vocos-22khz-univ.onnx \
  --tokens ./matcha-icefall-en_US-ljspeech/tokens.txt \
  --lexicon ./lexicon-en-us.txt \
  --sid 0 \
  --speed 1.0 \
  --output-wav matcha-en-1-lexicon.wav \
  --text "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone." \

ls -lh *.wav
