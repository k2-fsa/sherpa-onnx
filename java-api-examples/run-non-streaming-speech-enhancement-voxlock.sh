#!/usr/bin/env bash

set -ex

source ./setup.sh

if [ ! -d ./sherpa-onnx-voxlock-int8 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2
  tar xvf sherpa-onnx-voxlock-int8.tar.bz2
  rm sherpa-onnx-voxlock-int8.tar.bz2
fi

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  NonStreamingSpeechEnhancementVoxLock.java

mkdir -p NonStreamingSpeechEnhancementVoxLock
mv voxlock-enhanced-16k.wav NonStreamingSpeechEnhancementVoxLock/
cp sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav NonStreamingSpeechEnhancementVoxLock/
