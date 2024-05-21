#!/usr/bin/env bash

set -ex

if [ ! -d ./sherpa-onnx-streaming-paraformer-bilingual-zh-en ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
  tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
  rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
fi

go mod tidy
go build

./streaming-decode-files \
  --paraformer-encoder ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx \
  --paraformer-decoder ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx \
  --tokens ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt \
  --decoding-method greedy_search \
  --model-type paraformer \
  --debug 0 \
  ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/0.wav
