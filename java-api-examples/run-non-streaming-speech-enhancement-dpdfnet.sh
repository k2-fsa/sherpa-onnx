#!/usr/bin/env bash

set -ex

source ./setup.sh

if [ ! -f ./dpdfnet_baseline.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet_baseline.onnx
fi

if [ ! -f ./inp_16k.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav
fi

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  NonStreamingSpeechEnhancementDpdfNet.java
