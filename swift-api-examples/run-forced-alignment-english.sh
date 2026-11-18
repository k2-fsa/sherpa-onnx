#!/usr/bin/env bash

set -ex

if [ ! -d ../build-macos ]; then
  echo "Please run ../build-macos.sh first!"
  exit 1
fi

# please visit
# https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models
# to download more models
if [ ! -f ./english-mfa-v3.1.0/lexicon.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2
  tar xf english-mfa-v3.1.0.tar.bz2
  rm english-mfa-v3.1.0.tar.bz2
fi

if [ ! -e ./forced-alignment-english ] || [ ../build-macos/install/lib/libsherpa-onnx-c-api.a -nt ./forced-alignment-english ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./forced-alignment-english.swift  ./SherpaOnnx.swift \
    -L ../build-macos/install/lib/ \
    -l sherpa-onnx-c-api \
    -l onnxruntime \
    -o ./forced-alignment-english

  strip ./forced-alignment-english
else
  echo "./forced-alignment-english exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-macos/install/lib:$DYLD_LIBRARY_PATH
./forced-alignment-english
