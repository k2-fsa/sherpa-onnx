#!/usr/bin/env bash

set -ex

if [[ ! -f ../build/lib/libsherpa-onnx-jni.dylib  && ! -f ../build/lib/libsherpa-onnx-jni.so ]]; then
  mkdir -p ../build
  pushd ../build
  cmake \
    -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
    -DSHERPA_ONNX_ENABLE_TESTS=OFF \
    -DSHERPA_ONNX_ENABLE_CHECK=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_ONNX_ENABLE_JNI=ON \
    ..

  make -j4
  ls -lh lib
  popd
fi

if [ ! -f ../sherpa-onnx/java-api/build/sherpa-onnx.jar ]; then
  pushd ../sherpa-onnx/java-api
  make
  popd
fi

# please visit
# https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
# to download more models
if [ ! -f ./vits-zh-hf-fanchen-C/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-hf-fanchen-C.tar.bz2
  tar xf vits-zh-hf-fanchen-C.tar.bz2
  rm vits-zh-hf-fanchen-C.tar.bz2
fi

java \
  -Djava.library.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/build/sherpa-onnx.jar \
  NonStreamingTtsVitsZh.java
