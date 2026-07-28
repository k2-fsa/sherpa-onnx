#!/usr/bin/env bash

set -ex

source ./setup.sh

# please visit
# https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kitten.html
# to download more models

if [ ! -f ./kitten-nano-en-v0_1-fp16/model.fp16.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_1-fp16.tar.bz2
  tar xf kitten-nano-en-v0_1-fp16.tar.bz2
  rm kitten-nano-en-v0_1-fp16.tar.bz2
fi

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  NonStreamingTtsKittenEn.java
