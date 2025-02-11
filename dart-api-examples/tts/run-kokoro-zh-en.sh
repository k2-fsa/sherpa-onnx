#!/usr/bin/env bash

set -ex

dart pub get

# please visit
# https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html
# to download more models
if [ ! -f ./kokoro-multi-lang-v1_0/model.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
  tar xf kokoro-multi-lang-v1_0.tar.bz2
  rm kokoro-multi-lang-v1_0.tar.bz2
fi

dart run \
  ./bin/kokoro-zh-en.dart \
  --model ./kokoro-multi-lang-v1_0/model.onnx \
  --voices ./kokoro-multi-lang-v1_0/voices.bin \
  --tokens ./kokoro-multi-lang-v1_0/tokens.txt \
  --data-dir ./kokoro-multi-lang-v1_0/espeak-ng-data \
  --dict-dir ./kokoro-multi-lang-v1_0/dict \
  --lexicon ./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
  --sid 45 \
  --speed 1.0 \
  --output-wav kokoro-zh-en-45.wav \
  --text "中英文语音合成测试。This is generated by next generation Kaldi using Kokoro without Misaki. 你觉得中英文说的如何呢？"

ls -lh *.wav
