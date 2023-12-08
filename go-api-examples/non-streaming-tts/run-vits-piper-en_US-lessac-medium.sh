#!/usr/bin/env bash

# please refer to
# https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#en-us-lessac-medium-english-single-speaker
# to download the model before you run this script

./non-streaming-tts \
  --vits-model=./vits-piper-en_US-lessac-medium/en_US-lessac-medium.onnx \
  --vits-data-dir=./vits-piper-en_US-lessac-medium/espeak-ng-data \
  --vits-tokens=./vits-piper-en_US-lessac-medium/tokens.txt \
  --output-filename=./liliana-piper-en_US-lessac-medium.wav \
  'liliana, the most beautiful and lovely assistant of our team!'
