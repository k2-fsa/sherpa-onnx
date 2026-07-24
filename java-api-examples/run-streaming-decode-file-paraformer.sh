#!/usr/bin/env bash

set -ex

source ./setup.sh

if [ ! -f ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
  tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
  rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
fi

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  StreamingDecodeFileParaformer.java
