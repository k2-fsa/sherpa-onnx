#!/usr/bin/env bash

set -ex

if [ ! -d ../build-swift-macos ]; then
  echo "Please run ../build-swift-macos.sh first!"
  exit 1
fi

if [ ! -d ./sherpa-onnx-whisper-tiny.en ]; then
  echo "Please download the pre-trained model for testing."
  echo "You can refer to"
  echo ""
  echo "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html"
  echo ""
  echo "for help"

  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
  tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
  rm sherpa-onnx-whisper-tiny.en.tar.bz2
  ls -lh sherpa-onnx-whisper-tiny.en
fi
if [ ! -f ./silero_vad.onnx ]; then
  echo "downloading silero_vad"
  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
fi

if [ ! -e ./generate-subtitles ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-swift-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./generate-subtitles.swift  ./SherpaOnnx.swift \
    -L ../build-swift-macos/install/lib/ \
    -l sherpa-onnx \
    -l onnxruntime \
    -o generate-subtitles

  strip generate-subtitles
else
  echo "./generate-subtitles exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-swift-macos/install/lib:$DYLD_LIBRARY_PATH
./generate-subtitles
