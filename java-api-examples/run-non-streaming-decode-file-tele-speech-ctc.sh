#!/usr/bin/env bash

set -ex

source ./setup.sh

if [ ! -f ./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2
  tar xvf sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2
  rm sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2
fi

java \
  -Djava.library.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  ./NonStreamingDecodeFileTeleSpeechCtc.java
