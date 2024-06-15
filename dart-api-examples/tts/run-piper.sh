#!/usr/bin/env bash

set -ex

dart pub get


# Please visit
# https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
# to download more models

if [[ ! -f ./vits-piper-en_US-libritts_r-medium/tokens.txt ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2
  tar xf vits-piper-en_US-libritts_r-medium.tar.bz2
  rm vits-piper-en_US-libritts_r-medium.tar.bz2
fi

dart run \
  ./bin/piper.dart \
  --model ./vits-piper-en_US-libritts_r-medium/en_US-libritts_r-medium.onnx \
  --tokens ./vits-piper-en_US-libritts_r-medium/tokens.txt \
  --data-dir ./vits-piper-en_US-libritts_r-medium/espeak-ng-data \
  --sid 351 \
  --speed 1.0 \
  --text 'How are you doing? This is a speech to text example, using next generation kaldi with piper.' \
  --output-wav piper-351.wav

ls -lh *.wav
