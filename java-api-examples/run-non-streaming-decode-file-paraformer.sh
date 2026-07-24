#!/usr/bin/env bash

set -ex

source ./setup.sh

if [ ! -f ./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

  tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
fi

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  NonStreamingDecodeFileParaformer.java
