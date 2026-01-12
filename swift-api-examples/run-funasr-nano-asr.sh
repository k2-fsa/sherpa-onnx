#!/usr/bin/env bash

set -ex

if [ ! -d ../build-swift-macos ]; then
  echo "Please run ../build-swift-macos.sh first!"
  exit 1
fi

if [ ! -f ./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
  tar xvf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
  rm sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
fi

if [ ! -e ./funasr-nano ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-swift-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./funasr-nano.swift  ./SherpaOnnx.swift \
    -L ../build-swift-macos/install/lib/ \
    -l sherpa-onnx \
    -l onnxruntime \
    -o funasr-nano

  strip funasr-nano
else
  echo "./funasr-nano exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-swift-macos/install/lib:$DYLD_LIBRARY_PATH
./funasr-nano

