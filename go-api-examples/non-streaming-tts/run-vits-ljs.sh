#!/usr/bin/env bash

# please refer to
# https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#ljspeech-english-single-speaker
# to download the model before you run this script

./non-streaming-tts \
  --vits-model=./vits-ljs/vits-ljs.onnx \
  --vits-lexicon=./vits-ljs/lexicon.txt \
  --vits-tokens=./vits-ljs/tokens.txt \
  --sid=0 \
  --debug=1 \
  --output-filename=./vits-ljs.wav \
  "Liliana, the most beautiful and lovely assistant of our team!"
