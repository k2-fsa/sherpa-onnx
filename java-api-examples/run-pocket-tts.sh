#!/usr/bin/env bash

set -ex

source ./setup.sh

# please visit
# https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
# to download more models

if [ ! -f ./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
  tar xvf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
  rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
fi

if false; then
  javac \
    -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
    PocketTts.java
  javap -p -s PocketTts.class
  javap -p -s PocketTts$1.class
fi

java \
  -Djava.library.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  PocketTts.java
