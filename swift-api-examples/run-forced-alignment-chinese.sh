#!/usr/bin/env bash

set -ex

if [ ! -d ../build-macos ]; then
  echo "Please run ../build-macos.sh first!"
  exit 1
fi

# please visit
# https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models
# to download more models
if [ ! -f ./zh-aishell3-pinyin/lexicon.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2
  tar xf zh-aishell3-pinyin.tar.bz2
  rm zh-aishell3-pinyin.tar.bz2
fi

if [ ! -e ./forced-alignment-chinese ] || [ ../build-macos/install/lib/libsherpa-onnx-c-api.a -nt ./forced-alignment-chinese ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./forced-alignment-chinese.swift  ./SherpaOnnx.swift \
    -L ../build-macos/install/lib/ \
    -l sherpa-onnx-c-api \
    -l onnxruntime \
    -o ./forced-alignment-chinese

  strip ./forced-alignment-chinese
else
  echo "./forced-alignment-chinese exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-macos/install/lib:$DYLD_LIBRARY_PATH
./forced-alignment-chinese
