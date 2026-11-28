#!/usr/bin/env bash

set -ex

if [ ! -d ../build-macos ]; then
  echo "Please run ../build-macos.sh first!"
  exit 1
fi

if [ ! -f ./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/model.int8.onnx ]; then
  echo "Please download the pre-trained model for testing."
  echo "You can refer to"
  echo ""
  echo "https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models"
  echo ""
  echo "for help"

  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2
  tar xvf sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2
  rm sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2

  ls -lh sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20
fi

if [ ! -e ./w2v-bert-ctc ] || [ ../build-macos/install/lib/libsherpa-onnx-c-api.a -nt ./w2v-bert-ctc ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./w2v-bert-ctc.swift  ./SherpaOnnx.swift \
    -L ../build-macos/install/lib/ \
    -l sherpa-onnx-c-api \
    -l onnxruntime \
    -o w2v-bert-ctc

  strip w2v-bert-ctc
else
  echo "./w2v-bert-ctc exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-macos/install/lib:$DYLD_LIBRARY_PATH
./w2v-bert-ctc
