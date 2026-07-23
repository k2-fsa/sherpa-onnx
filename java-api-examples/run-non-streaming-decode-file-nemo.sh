#!/usr/bin/env bash

set -ex

source ./setup.sh

if [ ! -f ./sherpa-onnx-nemo-ctc-en-citrinet-512/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
  tar xvf sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
  rm sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
fi

java \
  -Djava.library.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  NonStreamingDecodeFileNemo.java
