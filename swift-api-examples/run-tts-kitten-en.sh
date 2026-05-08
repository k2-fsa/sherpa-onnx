#!/usr/bin/env bash

set -ex

if [ ! -d ../build-swift-macos ]; then
  echo "Please run ../build-swift-macos.sh first!"
  exit 1
fi

# please visit
# https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kitten.html
# to download more models
if [ ! -f ./kitten-mini-en-v0_8/model.onnx ]; then
  echo "Please generate or copy kitten-mini-en-v0_8 first."
  echo "For example:"
  echo "  cd ../scripts/kitten-tts/v0_8"
  echo "  ./run.sh KittenML/kitten-tts-mini-0.8"
  echo "  cd ../../../swift-api-examples"
  echo "  cp -R ../scripts/kitten-tts/v0_8 ./kitten-mini-en-v0_8"
  exit 1
fi

if [ ! -d ./kitten-mini-en-v0_8/espeak-ng-data ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2
  tar xf espeak-ng-data.tar.bz2
  rm espeak-ng-data.tar.bz2
  mv espeak-ng-data ./kitten-mini-en-v0_8/
fi

if [ ! -e ./tts-kitten-en ] || [ ./tts-kitten-en.swift -nt ./tts-kitten-en ] || [ ./SherpaOnnx.swift -nt ./tts-kitten-en ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-swift-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./tts-kitten-en.swift  ./SherpaOnnx.swift \
    -L ../build-swift-macos/install/lib/ \
    -l sherpa-onnx \
    -l onnxruntime \
    -o tts-kitten-en

  strip tts-kitten-en
else
  echo "./tts-kitten-en exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-swift-macos/install/lib:$DYLD_LIBRARY_PATH
./tts-kitten-en
