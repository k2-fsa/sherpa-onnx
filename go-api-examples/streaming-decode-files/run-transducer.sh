#!/usr/bin/env bash

set -ex

if [ ! -d sherpa-onnx-streaming-zipformer-en-2023-06-26 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
  tar xvf sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
  rm sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
fi

go mod tidy
go build

./streaming-decode-files \
  --encoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx \
  --decoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx \
  --joiner ./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx \
  --tokens ./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt \
  --model-type zipformer2 \
  --debug 0 \
  ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav
