#!/usr/bin/env bash
set -ex

# see
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-2023-06-21-english
if [ ! -f ./sherpa-onnx-streaming-zipformer-en-2023-06-21/encoder-epoch-99-avg-1.int8.onnx ]; then
  curl -SsL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2

  tar xvf sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2
  rm sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2
  ls -lh sherpa-onnx-streaming-zipformer-en-2023-06-21
fi

cargo run --example streaming_zipformer -- \
    --wav sherpa-onnx-streaming-zipformer-en-2023-06-21/test_wavs/1.wav \
    --encoder sherpa-onnx-streaming-zipformer-en-2023-06-21/encoder-epoch-99-avg-1.int8.onnx \
    --decoder sherpa-onnx-streaming-zipformer-en-2023-06-21/decoder-epoch-99-avg-1.onnx \
    --joiner sherpa-onnx-streaming-zipformer-en-2023-06-21/joiner-epoch-99-avg-1.int8.onnx \
    --tokens sherpa-onnx-streaming-zipformer-en-2023-06-21/tokens.txt \
    --provider cpu \
    --debug
