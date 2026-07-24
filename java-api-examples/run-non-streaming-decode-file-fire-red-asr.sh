#!/usr/bin/env bash

set -ex

source ./setup.sh

if [ ! -f ./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
  tar xvf sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
  rm sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
  ls -lh sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16
fi

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  NonStreamingDecodeFileFireRedAsr.java
