#!/usr/bin/env bash
# Common setup script for sherpa-onnx Java API examples
# Source this file at the beginning of each run-xxx.sh script:
#   source ./setup.sh

# Build sherpa-onnx C++ library if not exists
if [[ ! -f ../build/lib/libsherpa-onnx-jni.dylib && ! -f ../build/lib/libsherpa-onnx-jni.so && ! -f ../build/lib/sherpa-onnx-jni.dll ]]; then
  mkdir -p ../build
  pushd ../build
  cmake \
    -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
    -DSHERPA_ONNX_ENABLE_TESTS=OFF \
    -DSHERPA_ONNX_ENABLE_CHECK=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_ONNX_ENABLE_JNI=ON \
    -DCMAKE_INSTALL_PREFIX=./install \
    ..

  cmake --build . --config Release
  cmake --build . --config Release --target install
  popd

  # On Windows, copy DLLs to build/lib so scripts can find them
  if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* || "$(uname -s)" == CYGWIN* ]]; then
    mkdir -p ../build/lib
    cp -v ../build/install/lib/*.dll ../build/lib/ 2>/dev/null || true
    cp -v ../build/install/lib/*.lib ../build/lib/ 2>/dev/null || true
  fi
fi

# Build sherpa-onnx JVM jar if not exists
if [ ! -f ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar ]; then
  pushd ../sherpa-onnx/java-api
  mvn package
  popd
fi
