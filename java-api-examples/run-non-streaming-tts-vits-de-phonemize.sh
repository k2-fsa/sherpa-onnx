#!/usr/bin/env bash

set -ex

source ./setup.sh

# please visit
# https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
# to download more models
#
# There is no German lexicon, so this model is phonemize-only.
if [ ! -f ./vits-piper-de_DE-glados-low/de_DE-glados-low.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-de_DE-glados-low.tar.bz2
  tar xf vits-piper-de_DE-glados-low.tar.bz2
  rm vits-piper-de_DE-glados-low.tar.bz2
fi

source ./setup-phonemize.sh

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp "$PHONEMIZE_CLASSPATH" \
  NonStreamingTtsVitsDePhonemize.java
