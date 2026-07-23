#!/usr/bin/env bash

set -ex

source ./setup.sh

if [ ! -f ./sherpa-onnx-zipformer-gigaspeech-2023-12-12/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2

  tar xvf sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2
  rm sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2
fi

java \
  -Djava.library.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  NonStreamingDecodeFileTransducer.java
