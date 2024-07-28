#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-zipformer-gigaspeech-2023-12-12/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2

  tar xvf sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2
  rm sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2
fi

if [ ! -f ./Obama.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
fi

if [[ ! -f ./silero_vad.onnx ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
fi

dart run \
  ./bin/zipformer-transducer.dart \
  --silero-vad ./silero_vad.onnx \
  --encoder ./sherpa-onnx-zipformer-gigaspeech-2023-12-12/encoder-epoch-30-avg-1.int8.onnx \
  --decoder ./sherpa-onnx-zipformer-gigaspeech-2023-12-12/decoder-epoch-30-avg-1.onnx \
  --joiner ./sherpa-onnx-zipformer-gigaspeech-2023-12-12/joiner-epoch-30-avg-1.int8.onnx \
  --tokens ./sherpa-onnx-zipformer-gigaspeech-2023-12-12/tokens.txt \
  --input-wav ./Obama.wav

