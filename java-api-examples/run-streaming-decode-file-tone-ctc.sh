#!/usr/bin/env bash

set -ex

source ./setup.sh

if [ ! -f ./sherpa-onnx-streaming-t-one-russian-2025-09-08/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
  tar xvf sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
  rm sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
fi

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  StreamingDecodeFileToneCtc.java
