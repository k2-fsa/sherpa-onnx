#!/usr/bin/env bash

set -ex

dart pub get


# Please visit
# https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
# to download more models

if [[ ! -f ./vits-piper-de_DE-glados-low/tokens.txt ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-de_DE-glados-low.tar.bz2
  tar xf vits-piper-de_DE-glados-low.tar.bz2
  rm vits-piper-de_DE-glados-low.tar.bz2
fi

dart run \
  ./bin/vits-de-phonemize.dart \
  --model ./vits-piper-de_DE-glados-low/de_DE-glados-low.onnx \
  --tokens ./vits-piper-de_DE-glados-low/tokens.txt \
  --sid 0 \
  --speed 1.0 \
  --text 'Wie geht es dir? Das ist ein Text-zu-Sprache-Beispiel mit next generation Kaldi.' \
  --output-wav vits-de-0-phonemize.wav

ls -lh *.wav
