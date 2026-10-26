#!/usr/bin/env bash

set -ex

if [ ! -d ../build-macos ]; then
  echo "Please run ../build-macos.sh first!"
  exit 1
fi

if [ ! -d ./sherpa-onnx-voxlock-int8 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2
  tar xvf sherpa-onnx-voxlock-int8.tar.bz2
  rm sherpa-onnx-voxlock-int8.tar.bz2
fi

if [ ! -e ./online-speech-enhancement-voxlock ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./online-speech-enhancement-voxlock.swift ./SherpaOnnx.swift \
    -L ../build-macos/install/lib/ \
    -l sherpa-onnx-c-api \
    -l onnxruntime \
    -o online-speech-enhancement-voxlock

  strip online-speech-enhancement-voxlock
else
  echo "./online-speech-enhancement-voxlock  exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-macos/install/lib:$DYLD_LIBRARY_PATH
./online-speech-enhancement-voxlock

mkdir -p wav/online-speech-enhancement-voxlock
mv voxlock-enhanced-online-16k.wav wav/online-speech-enhancement-voxlock/
cp sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav wav/online-speech-enhancement-voxlock/
