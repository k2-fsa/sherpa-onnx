#!/usr/bin/env bash

set -ex

if [ ! -d ../build-swift-macos ]; then
  echo "Please run ../build-swift-macos.sh first!"
  exit 1
fi

if [ ! -f ./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/encoder.int8.onnx ]; then
  echo "Please download the pre-trained model for testing."
  echo "You can refer to"
  echo ""
  echo "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html"
  echo ""
  echo "for help"

  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2
  tar xvf sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2
  rm sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2

  ls -lh sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01
fi

if [ ! -e ./cohere-transcribe ]; then
  swiftc \
    -lc++ \
    -I ../build-swift-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./cohere-transcribe.swift ./SherpaOnnx.swift \
    -L ../build-swift-macos/install/lib/ \
    -l sherpa-onnx \
    -l onnxruntime \
    -o cohere-transcribe

  strip cohere-transcribe
else
  echo "./cohere-transcribe exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-swift-macos/install/lib:$DYLD_LIBRARY_PATH
./cohere-transcribe
