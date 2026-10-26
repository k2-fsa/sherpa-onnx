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
  StreamingSpeechEnhancementVoxLock.java

mkdir -p StreamingSpeechEnhancementVoxLock
mv voxlock-enhanced-online-16k.wav StreamingSpeechEnhancementVoxLock/
cp sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav StreamingSpeechEnhancementVoxLock/
