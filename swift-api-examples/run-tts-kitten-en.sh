#!/usr/bin/env bash

set -ex

if [ ! -d ../build-swift-macos ]; then
  echo "Please run ../build-swift-macos.sh first!"
  exit 1
fi

model_dir=./kitten-mini-en-v0_8

if [ ! -f ./kitten-mini-en-v0_8/model.onnx ] || \
   [ ! -f ./kitten-mini-en-v0_8/voices.bin ] || \
   [ ! -f ./kitten-mini-en-v0_8/tokens.txt ]; then
  src_dir=../scripts/kitten-tts/v0_8
  (
    cd "${src_dir}"
    if ! python3 - <<'PY'
import importlib.util
import sys

sys.exit(
    0
    if importlib.util.find_spec("numpy") and importlib.util.find_spec("onnx")
    else 1
)
PY
    then
      python3 -m venv .venv
      . .venv/bin/activate
      python3 -m pip install -q numpy onnx
    fi

    ./run.sh KittenML/kitten-tts-mini-0.8
  )
  rm -rf "${model_dir}"
  cp -R "${src_dir}/kitten-mini-en-v0_8" "${model_dir}"
fi

if [ ! -d "${model_dir}/espeak-ng-data" ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2
  tar xf espeak-ng-data.tar.bz2
  rm espeak-ng-data.tar.bz2
  mv espeak-ng-data "${model_dir}/"
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
