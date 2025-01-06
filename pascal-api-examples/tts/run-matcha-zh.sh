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
# https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html#matcha-icefall-zh-baker-chinese-1-female-speaker
# to download more models
if [ ! -f ./matcha-icefall-zh-baker/model-steps-3.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
  tar xvf matcha-icefall-zh-baker.tar.bz2
  rm matcha-icefall-zh-baker.tar.bz2
fi

if [ ! -f ./hifigan_v2.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx
fi

fpc \
  -dSHERPA_ONNX_USE_SHARED_LIBS \
  -Fu$SHERPA_ONNX_DIR/sherpa-onnx/pascal-api \
  -Fl$SHERPA_ONNX_DIR/build/install/lib \
  ./matcha-zh.pas

export LD_LIBRARY_PATH=$SHERPA_ONNX_DIR/build/install/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$SHERPA_ONNX_DIR/build/install/lib:$DYLD_LIBRARY_PATH

./matcha-zh
