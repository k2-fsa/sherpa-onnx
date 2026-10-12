#!/usr/bin/env bash

set -ex

if [ ! -d ../build-macos ]; then
  echo "Please run ../build-macos.sh first!"
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

if [ ! -e ./lexicon-en-us.txt ]; then
  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
fi

if [ ! -e ./tts-kitten-en-lexicon ] || [ ./tts-kitten-en-lexicon.swift -nt ./tts-kitten-en-lexicon ] || [ ./SherpaOnnx.swift -nt ./tts-kitten-en-lexicon ] || [ ../build-macos/install/lib/libsherpa-onnx-c-api.a -nt ./tts-kitten-en-lexicon ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./tts-kitten-en-lexicon.swift  ./SherpaOnnx.swift \
    -L ../build-macos/install/lib/ \
    -l sherpa-onnx-c-api \
    -l onnxruntime \
    -o tts-kitten-en-lexicon

  strip tts-kitten-en-lexicon
else
  echo "./tts-kitten-en-lexicon exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-macos/install/lib:$DYLD_LIBRARY_PATH
./tts-kitten-en-lexicon
