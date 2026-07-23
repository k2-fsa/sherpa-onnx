#!/usr/bin/env bash

set -ex

source ./setup.sh

if [ ! -f ./sherpa-onnx-conformer-zh-stateless2-2023-05-23/tokens.txt ]; then
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-conformer-zh-stateless2-2023-05-23.tar.bz2
  tar xvf sherpa-onnx-conformer-zh-stateless2-2023-05-23.tar.bz2
  rm sherpa-onnx-conformer-zh-stateless2-2023-05-23.tar.bz2
fi

if [ ! -f hotwords_cn.txt ]; then
  cat > hotwords_cn.txt <<EOF
朱丽楠
EOF
fi

java \
  -Djava.library.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  NonStreamingDecodeFileTransducerHotwords.java
