#!/usr/bin/env bash

set -ex

if [ ! -d ../build-macos ]; then
  echo "Please run ../build-macos.sh first!"
  exit 1
fi

# please visit
# https://k2-fsa.github.io/sherpa/onnx/tts/zipvoice.html
# to download more models
if [ ! -f ./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
  tar xf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
  rm sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
fi

if [ ! -f ./vocos_24khz.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx
fi

if [ ! -e ./lexicon-en-us.txt ]; then
  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
fi

if [ ! -e ./tts-zipvoice-lexicon ]; then
  swiftc \
    -lc++ \
    -I ../build-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./tts-zipvoice-lexicon.swift ./SherpaOnnx.swift \
    -L ../build-macos/install/lib/ \
    -l sherpa-onnx-c-api \
    -l onnxruntime \
    -o tts-zipvoice-lexicon

  strip tts-zipvoice-lexicon
else
  echo "./tts-zipvoice-lexicon exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-macos/install/lib:$DYLD_LIBRARY_PATH
./tts-zipvoice-lexicon
