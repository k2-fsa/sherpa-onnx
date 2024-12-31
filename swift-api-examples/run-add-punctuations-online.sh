#!/usr/bin/env bash

set -ex

if [ ! -d ../build-swift-macos ]; then
  echo "Please run ../build-swift-macos.sh first!"
  exit 1
fi

# Download and extract the online punctuation model if not exists
if [ ! -d ./sherpa-onnx-online-punct-en-2024-08-06 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
  tar xvf sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
  rm sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
fi

if [ ! -e ./add-punctuation-online ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-swift-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./add-punctuation-online.swift  ./SherpaOnnx.swift \
    -L ../build-swift-macos/install/lib/ \
    -l sherpa-onnx \
    -l onnxruntime \
    -o ./add-punctuation-online

  strip ./add-punctuation-online
else
  echo "./add-punctuation-online exists - skip building"
fi

# Set library path and run the executable
export DYLD_LIBRARY_PATH=$PWD/../build-swift-macos/install/lib:$DYLD_LIBRARY_PATH
./add-punctuation-online 