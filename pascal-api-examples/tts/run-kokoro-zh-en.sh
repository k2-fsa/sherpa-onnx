#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SHERPA_ONNX_DIR=$(cd $SCRIPT_DIR/../.. && pwd)

echo "SHERPA_ONNX_DIR: $SHERPA_ONNX_DIR"

if [[ ! -f ../../build/install/lib/libsherpa-onnx-c-api.dylib  && ! -f ../../build/install/lib/libsherpa-onnx-c-api.so && ! -f ../../build/install/lib/sherpa-onnx-c-api.dll ]]; then
  mkdir -p ../../build
  pushd ../../build
  cmake \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
    -DSHERPA_ONNX_ENABLE_TESTS=OFF \
    -DSHERPA_ONNX_ENABLE_CHECK=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
    ..

  cmake --build . --target install --config Release
  popd
fi

# please visit
# https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html
if [ ! -f ./kokoro-multi-lang-v1_0/model.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
  tar xf kokoro-multi-lang-v1_0.tar.bz2
  rm kokoro-multi-lang-v1_0.tar.bz2
fi

fpc \
  -dSHERPA_ONNX_USE_SHARED_LIBS \
  -Fu$SHERPA_ONNX_DIR/sherpa-onnx/pascal-api \
  -Fl$SHERPA_ONNX_DIR/build/install/lib \
  ./kokoro-zh-en.pas

export LD_LIBRARY_PATH=$SHERPA_ONNX_DIR/build/install/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$SHERPA_ONNX_DIR/build/install/lib:$DYLD_LIBRARY_PATH

./kokoro-zh-en
