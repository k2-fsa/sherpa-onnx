#!/usr/bin/env bash

set -ex

source ./setup.sh

# please visit
# https://k2-fsa.github.io/sherpa/onnx/tts/zipvoice.html
# to download more models
if [ ! -f ./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
  tar xf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
  rm sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
fi

if [ ! -f ./vocos_24khz.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx
fi

java \
  -Djava.library.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  ZipVoiceTts.java
