#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
  tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
  rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
fi

dart run \
  ./bin/zipformer-ctc-hlg.dart \
  --model ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx \
  --hlg ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst \
  --tokens ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/tokens.txt \
  --input-wav ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/test_wavs/1.wav
