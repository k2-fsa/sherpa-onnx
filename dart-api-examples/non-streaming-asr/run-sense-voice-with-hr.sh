#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
fi

if [ ! -d dict ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2
  tar xf dict.tar.bz2
  rm dict.tar.bz2

  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/test-hr.wav
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt
fi

dart run \
  ./bin/sense-voice-with-hr.dart \
  --model ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
  --tokens ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
  --use-itn true \
  --hr-dict-dir ./dict \
  --hr-lexicon ./lexicon.txt \
  --hr-rule-fsts ./replace.fst \
  --input-wav ./test-hr.wav
