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
  -cp ../sherpa-onnx/java-api/build/sherpa-onnx.jar \
  NonStreamingDecodeFileTransducerHotwords.java
