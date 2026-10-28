#!/usr/bin/env bash

set -ex

if [ ! -d ../build-macos ]; then
  echo "Please run ../build-macos.sh first!"
  exit 1
fi

if [ ! -f ./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx ]; then
  echo "Please download the pre-trained model for testing."
  echo "You can refer to"
  echo ""
  echo "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/icefall/zipformer.html#sherpa-onnx-zipformer-ctc-en-2023-10-02-english"
  echo ""
  echo "for help"

  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2

  tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
  rm sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
  ls -lh sherpa-onnx-zipformer-ctc-en-2023-10-02
fi

if [ ! -e ./zipformer-ctc-hlg ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./zipformer-ctc-hlg.swift  ./SherpaOnnx.swift \
    -L ../build-macos/install/lib/ \
    -l sherpa-onnx-c-api \
    -l onnxruntime \
    -o zipformer-ctc-hlg

  strip zipformer-ctc-hlg
else
  echo "./zipformer-ctc-hlg exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-macos/install/lib:$DYLD_LIBRARY_PATH
./zipformer-ctc-hlg
