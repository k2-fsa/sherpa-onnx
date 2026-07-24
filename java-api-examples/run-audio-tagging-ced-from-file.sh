#!/usr/bin/env bash

set -ex

source ./setup.sh

if [ ! -f ./sherpa-onnx-ced-mini-audio-tagging-2024-04-19/model.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2
  tar xvf sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2
  rm sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2
fi

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  ./AudioTaggingCEDFromFile.java
