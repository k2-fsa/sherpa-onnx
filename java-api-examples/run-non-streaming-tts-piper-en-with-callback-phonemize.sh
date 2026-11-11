#!/usr/bin/env bash

set -ex

source ./setup.sh

# please visit
# https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
# to download more models
if [ ! -f ./vits-piper-en_GB-cori-medium/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-cori-medium.tar.bz2
  tar xf vits-piper-en_GB-cori-medium.tar.bz2
  rm vits-piper-en_GB-cori-medium.tar.bz2
fi


source ./setup-phonemize.sh

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp "$PHONEMIZE_CLASSPATH" \
  NonStreamingTtsPiperEnWithCallbackPhonemize.java
