#!/usr/bin/env bash

set -ex

if [ ! -d ../build-macos ]; then
  echo "Please run ../build-macos.sh first!"
  exit 1
fi

if [ ! -d ./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
  tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
  rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
fi

if [ ! -e ./add-punctuations ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./add-punctuations.swift  ./SherpaOnnx.swift \
    -L ../build-macos/install/lib/ \
    -l sherpa-onnx-c-api \
    -l onnxruntime \
    -o ./add-punctuations

  strip ./add-punctuations
else
  echo "./add-punctuations exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-macos/install/lib:$DYLD_LIBRARY_PATH
./add-punctuations
