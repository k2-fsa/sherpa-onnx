#!/usr/bin/env bash

set -ex

source ./setup.sh

# please visit
# https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
# to download more models
if [ ! -f ./vits-coqui-de-css10/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-coqui-de-css10.tar.bz2
  tar xf vits-coqui-de-css10.tar.bz2
  rm vits-coqui-de-css10.tar.bz2
fi

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  NonStreamingTtsCoquiDe.java
