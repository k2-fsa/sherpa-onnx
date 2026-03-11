#!/usr/bin/env bash

set -ex

if [ ! -d ../build-swift-macos ]; then
  echo "Please run ../build-swift-macos.sh first!"
  exit 1
fi

# please visit
# https://k2-fsa.github.io/sherpa/onnx/tts/supertonic.html
# to download more models
if [ ! -f ./sherpa-onnx-supertonic-tts-int8-2026-03-06/duration_predictor.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
  tar xf sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
  rm sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
fi

if [ ! -e ./tts-supertonic-en ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-swift-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./tts-supertonic-en.swift  ./SherpaOnnx.swift \
    -L ../build-swift-macos/install/lib/ \
    -l sherpa-onnx \
    -l onnxruntime \
    -o tts-supertonic-en

  strip tts-supertonic-en
else
  echo "./tts-supertonic-en exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-swift-macos/install/lib:$DYLD_LIBRARY_PATH
./tts-supertonic-en
