#!/usr/bin/env bash

set -ex

source ./setup.sh

if [ ! -f ./silero_vad.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
fi

if [ ! -f ./sherpa-onnx-moonshine-tiny-en-int8/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
  tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
  rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
fi

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  ./VadFromMicWithNonStreamingMoonshine.java
