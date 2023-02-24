#!/usr/bin/env bash

set -ex

if [ ! -d ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 ]; then
  echo "Please download the pre-trained model for testing."
  echo "You can refer to"
  echo ""
  echo "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/zipformer-transducer-models.html#sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english"
  echo "for help"
  exit 1
fi

if [[ ! -f ../build/lib/libsherpa-onnx-core.a && ! -f ../build/lib/libsherpa-onnx-core.dylib  && ! -f ../build/lib/libsherpa-onnx-core.so ]]; then
  echo "Please build sherpa-onnx first. You can use"
  echo ""
  echo "  cd /path/to/sherpa-onnx"
  echo "  mkdir build"
  echo "  cd build"
  echo "  cmake .."
  echo "  make -j4"
  exit 1
fi

if [ ! -f ./decode-file-c-api ]; then
  make
fi

./decode-file-c-api \
  ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
  ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
  ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
  ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx \
  ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav
