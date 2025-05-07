#!/usr/bin/env bash

set -ex

if [ ! -d ../build-swift-macos ]; then
  echo "Please run ../build-swift-macos.sh first!"
  exit 1
fi

if [ ! -d ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
fi

if [ ! -d dict ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2
  tar xf dict.tar.bz2
  rm -rf dict.tar.bz2

  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/test-hr.wav
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt
fi

if [ ! -e ./decode-file-sense-voice-with-hr ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-swift-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./decode-file-sense-voice-with-hr.swift  ./SherpaOnnx.swift \
    -L ../build-swift-macos/install/lib/ \
    -l sherpa-onnx \
    -l onnxruntime \
    -o decode-file-sense-voice-with-hr

  strip decode-file-sense-voice-with-hr
else
  echo "./decode-file-sense-voice-with-hr exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-swift-macos/install/lib:$DYLD_LIBRARY_PATH
./decode-file-sense-voice-with-hr
