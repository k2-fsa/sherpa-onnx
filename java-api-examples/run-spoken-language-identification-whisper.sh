#!/usr/bin/env bash

set -ex

source ./setup.sh

# Note that it needs a multilingual whisper model. so, for example, tiny works while tiny.en does not work
# https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
if [ ! -f ./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
  tar xvf sherpa-onnx-whisper-tiny.tar.bz2
  rm sherpa-onnx-whisper-tiny.tar.bz2
fi

if [ ! -f ./spoken-language-identification-test-wavs/en-english.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/spoken-language-identification-test-wavs.tar.bz2
  tar xvf spoken-language-identification-test-wavs.tar.bz2
  rm spoken-language-identification-test-wavs.tar.bz2
fi

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  ./SpokenLanguageIdentificationWhisper.java
