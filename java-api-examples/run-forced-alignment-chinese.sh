#!/usr/bin/env bash

set -ex

source ./setup.sh

if [ ! -d ./zh-aishell3-pinyin ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2
  tar xvf zh-aishell3-pinyin.tar.bz2
  rm zh-aishell3-pinyin.tar.bz2
fi

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  ForcedAlignmentChinese.java

mkdir -p ForcedAlignmentChinese
mv zh.TextGrid ForcedAlignmentChinese/
cp zh-aishell3-pinyin/zh.wav ForcedAlignmentChinese/
