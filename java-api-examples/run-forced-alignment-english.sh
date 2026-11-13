#!/usr/bin/env bash

set -e

source ./setup.sh

if [ ! -d ./english-mfa-v3.1.0 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2
  tar xvf english-mfa-v3.1.0.tar.bz2
  rm english-mfa-v3.1.0.tar.bz2
fi

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -Dstdout.encoding=UTF-8 -Dstderr.encoding=UTF-8 \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  ForcedAlignmentEnglish.java

mkdir -p ForcedAlignmentEnglish
mv en.TextGrid ForcedAlignmentEnglish/
cp english-mfa-v3.1.0/en.wav ForcedAlignmentEnglish/
