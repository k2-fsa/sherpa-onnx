#!/usr/bin/env bash
# Common setup script for sherpa-onnx Java API examples
# Source this file at the beginning of each run-xxx.sh script:
#   source ./setup.sh

# Build sherpa-onnx C++ library if not exists
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

# Build sherpa-onnx JVM jar if not exists
if [ ! -f ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar ]; then
  pushd ../sherpa-onnx/java-api
  mvn package
  popd
fi
