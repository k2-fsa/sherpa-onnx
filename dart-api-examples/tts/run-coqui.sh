#!/usr/bin/env bash

set -ex

dart pub get


# Please visit
# https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
# to download more models

if [[ ! -f ./vits-coqui-de-css10/tokens.txt ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-coqui-de-css10.tar.bz2
  tar xvf vits-coqui-de-css10.tar.bz2
  rm vits-coqui-de-css10.tar.bz2
fi

# It is a character-based TTS model, so there is no need to use a lexicon
dart run \
  ./bin/coqui.dart \
  --model ./vits-coqui-de-css10/model.onnx \
  --tokens ./vits-coqui-de-css10/tokens.txt \
  --sid 0 \
  --speed 0.7 \
  --text 'Alles hat ein Ende, nur die Wurst hat zwei.' \
  --output-wav coqui-0.wav

ls -lh *.wav
