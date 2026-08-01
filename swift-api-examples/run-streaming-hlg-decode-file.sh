#!/usr/bin/env bash

set -ex

if [ ! -d ../build-macos ]; then
  echo "Please run ../build-macos.sh first!"
  exit 1
fi

if [ ! -f ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst ]; then
  echo "Downloading the pre-trained model for testing."

  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
  tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
  rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
fi

if [ ! -e ./streaming-hlg-decode-file ] || [ ../build-macos/install/lib/libsherpa-onnx-c-api.a -nt ./streaming-hlg-decode-file ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./streaming-hlg-decode-file.swift  ./SherpaOnnx.swift \
    -L ../build-macos/install/lib/ \
    -l sherpa-onnx-c-api \
    -l onnxruntime \
    -o streaming-hlg-decode-file

  strip ./streaming-hlg-decode-file
else
  echo "./streaming-hlg-decode-file exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-macos/install/lib:$DYLD_LIBRARY_PATH
./streaming-hlg-decode-file
