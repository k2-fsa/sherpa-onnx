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

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  NonStreamingTtsPiperEnWithCallback.java
