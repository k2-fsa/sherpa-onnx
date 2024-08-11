#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SHERPA_ONNX_DIR=$(cd $SCRIPT_DIR/../.. && pwd)

echo "SHERPA_ONNX_DIR: $SHERPA_ONNX_DIR"

if [[ ! -f ../../build/lib/libsherpa-onnx-c-api.dylib  && ! -f ../../build/lib/libsherpa-onnx-c-api.so ]]; then
  mkdir -p ../../build
  pushd ../../build
  cmake \
    -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
    -DSHERPA_ONNX_ENABLE_TESTS=OFF \
    -DSHERPA_ONNX_ENABLE_CHECK=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
    ..

  make -j4 sherpa-onnx-c-api
  ls -lh lib
  popd
fi

if [ ! -f ./lei-jun-test.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
fi

fpc \
  -Fu$SHERPA_ONNX_DIR/sherpa-onnx/pascal-api \
  -Fl$SHERPA_ONNX_DIR/build/lib \
  ./main.pas

export LD_LIBRARY_PATH=$SHERPA_ONNX_DIR/build/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$SHERPA_ONNX_DIR/build/lib:$DYLD_LIBRARY_PATH

./main
