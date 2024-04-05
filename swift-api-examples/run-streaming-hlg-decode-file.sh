#!/usr/bin/env bash

set -ex

if [ ! -d ../build-swift-macos ]; then
  echo "Please run ../build-swift-macos.sh first!"
  exit 1
fi

if [ ! -f ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst ]; then
  echo "Downloading the pre-trained model for testing."

  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
  tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
  rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
fi

if [ ! -e ./streaming-hlg-decode-file ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-swift-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./streaming-hlg-decode-file.swift  ./SherpaOnnx.swift \
    -L ../build-swift-macos/install/lib/ \
    -l sherpa-onnx \
    -l onnxruntime \
    -o streaming-hlg-decode-file

  strip ./streaming-hlg-decode-file
else
  echo "./streaming-hlg-decode-file exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-swift-macos/install/lib:$DYLD_LIBRARY_PATH
./streaming-hlg-decode-file
