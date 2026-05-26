#!/usr/bin/env bash

set -ex

if [ ! -d ../build-swift-macos ]; then
  echo "Please run ../build-swift-macos.sh first!"
  exit 1
fi

if [ ! -f ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx ]; then
  echo "Please download the pre-trained model for testing."
  echo "You can refer to"
  echo ""
  echo "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html"
  echo ""
  echo "for help"

  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
  tar xvf sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
  rm sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2

  ls -lh sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25
fi

if [ ! -e ./qwen3-asr ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-swift-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./qwen3-asr.swift ./SherpaOnnx.swift \
    -L ../build-swift-macos/install/lib/ \
    -l sherpa-onnx \
    -l onnxruntime \
    -o qwen3-asr

  strip qwen3-asr
else
  echo "./qwen3-asr exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-swift-macos/install/lib:$DYLD_LIBRARY_PATH
./qwen3-asr
