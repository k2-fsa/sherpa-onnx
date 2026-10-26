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
  NonStreamingSpeechEnhancementVoxLockWithVoiceSample.java

mkdir -p NonStreamingSpeechEnhancementVoxLockWithVoiceSample
mv voxlock-enhanced-with-voice-sample-16k.wav NonStreamingSpeechEnhancementVoxLockWithVoiceSample/
cp sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav sherpa-onnx-voxlock-int8/test_wavs/en-voice-sample-f.wav NonStreamingSpeechEnhancementVoxLockWithVoiceSample/
