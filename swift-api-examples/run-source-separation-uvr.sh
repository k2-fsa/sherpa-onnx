#!/usr/bin/env bash

set -ex

if [ ! -d ../build-macos ]; then
  echo "Please run ../build-macos.sh first!"
  exit 1
fi

if [ ! -f ./UVR-MDX-NET-Voc_FT.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Voc_FT.onnx
fi

if [ ! -f ./qi-feng-le-zh.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav
fi

if [ ! -e ./source-separation-uvr ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./source-separation-uvr.swift ./SherpaOnnx.swift \
    -L ../build-macos/install/lib/ \
    -l sherpa-onnx-c-api \
    -l onnxruntime \
    -o source-separation-uvr

  strip source-separation-uvr
else
  echo "./source-separation-uvr exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-macos/install/lib:$DYLD_LIBRARY_PATH
./source-separation-uvr
