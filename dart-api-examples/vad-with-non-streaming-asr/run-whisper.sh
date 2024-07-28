#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2

  tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
  rm sherpa-onnx-whisper-tiny.en.tar.bz2
fi



if [ ! -f ./Obama.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
fi

if [[ ! -f ./silero_vad.onnx ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
fi

dart run \
  ./bin/whisper.dart \
  --silero-vad ./silero_vad.onnx \
  --encoder ./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx \
  --decoder ./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx \
  --tokens ./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt \
  --input-wav ./Obama.wav
