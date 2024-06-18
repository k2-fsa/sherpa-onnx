#!/usr/bin/env bash

set -ex

if [ ! -d ../build-swift-macos ]; then
  echo "Please run ../build-swift-macos.sh first!"
  exit 1
fi

if [ ! -d ./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
  tar xf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
  rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
fi

if [ ! -e ./keyword-spotting-from-file ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-swift-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./keyword-spotting-from-file.swift  ./SherpaOnnx.swift \
    -L ../build-swift-macos/install/lib/ \
    -l sherpa-onnx \
    -l onnxruntime \
    -o keyword-spotting-from-file

  strip keyword-spotting-from-file
else
  echo "./keyword-spotting-from-file exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-swift-macos/install/lib:$DYLD_LIBRARY_PATH
./keyword-spotting-from-file
