#!/usr/bin/env bash

set -ex

if [[ ! -f ../../build/install/lib/libsherpa-onnx-core.dylib && ! -f ../../build/install/lib/libsherpa-onnx-core.so ]]; then
  pushd ../../
  mkdir -p build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=./install -DBUILD_SHARED_LIBS=ON ..
  make install
  popd
fi
export SHERPA_ONNX_INSTALL_DIR=$PWD/../../build/install

./node_modules/.bin/cmake-js compile
