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

if [ ! -e ./speech-enhancement-voxlock-with-voice-sample ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./speech-enhancement-voxlock-with-voice-sample.swift ./SherpaOnnx.swift \
    -L ../build-macos/install/lib/ \
    -l sherpa-onnx-c-api \
    -l onnxruntime \
    -o speech-enhancement-voxlock-with-voice-sample

  strip speech-enhancement-voxlock-with-voice-sample
else
  echo "./speech-enhancement-voxlock-with-voice-sample  exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-macos/install/lib:$DYLD_LIBRARY_PATH
./speech-enhancement-voxlock-with-voice-sample

mkdir -p wav/speech-enhancement-voxlock-with-voice-sample
mv voxlock-enhanced-with-voice-sample-16k.wav wav/speech-enhancement-voxlock-with-voice-sample/
cp sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav sherpa-onnx-voxlock-int8/test_wavs/en-voice-sample-f.wav wav/speech-enhancement-voxlock-with-voice-sample/
