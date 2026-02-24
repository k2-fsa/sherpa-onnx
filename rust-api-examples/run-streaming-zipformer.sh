#!/usr/bin/env bash
set -ex

if [ ! -f ./sherpa-onnx-streaming-zipformer-en-2023-06-21/encoder-epoch-99-avg-1.int8.onnx ]; then
  curl -SsL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2

  tar xvf sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2
  rm sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2
  ls -lh sherpa-onnx-streaming-zipformer-en-2023-06-21
fi

cargo run --example streaming_zipformer
