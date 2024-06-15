#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
  tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
  rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
fi

dart run \
  ./bin/paraformer.dart \
  --encoder ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx \
  --decoder ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx \
  --tokens ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt \
  --input-wav ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/0.wav
