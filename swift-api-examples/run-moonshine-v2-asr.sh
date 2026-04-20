#!/usr/bin/env bash

set -ex

if [ ! -d ../build-swift-macos ]; then
  echo "Please run ../build-swift-macos.sh first!"
  exit 1
fi

if [ ! -f ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/encoder_model.ort ]; then
  echo "Please download the pre-trained model for testing."
  echo "You can refer to"
  echo ""
  echo "https://k2-fsa.github.io/sherpa/onnx/moonshine/index.html"
  echo ""
  echo "for help"

  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
  tar xvf sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
  rm sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
  ls -lh sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27
fi

if [ ! -e ./moonshine-v2-asr ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-swift-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./moonshine-v2-asr.swift  ./SherpaOnnx.swift \
    -L ../build-swift-macos/install/lib/ \
    -l sherpa-onnx \
    -l onnxruntime \
    -o moonshine-v2-asr

  strip moonshine-v2-asr
else
  echo "./moonshine-v2-asr exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-swift-macos/install/lib:$DYLD_LIBRARY_PATH
./moonshine-v2-asr
