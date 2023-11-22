#!/usr/bin/env bash

set -ex

if [ ! -d ../build-swift-macos ]; then
  echo "Please run ../build-swift-macos.sh first!"
  exit 1
fi

if [ ! -d ./sherpa-onnx-whisper-tiny.en ]; then
  echo "Please download the pre-trained model for testing."
  echo "You can refer to"
  echo ""
  echo "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html"
  echo ""
  echo "for help"

  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
  tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
  rm sherpa-onnx-whisper-tiny.en.tar.bz2
fi

if [ ! -e ./decode-file-non-streaming ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-swift-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./decode-file-non-streaming.swift  ./SherpaOnnx.swift \
    -L ../build-swift-macos/install/lib/ \
    -l sherpa-onnx \
    -l onnxruntime \
    -o decode-file-non-streaming

  strip decode-file-non-streaming
else
  echo "./decode-file-non-streaming exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-swift-macos/install/lib:$DYLD_LIBRARY_PATH
./decode-file-non-streaming
