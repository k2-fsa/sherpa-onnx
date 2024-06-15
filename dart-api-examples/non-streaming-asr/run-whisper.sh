#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2

  tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
  rm sherpa-onnx-whisper-tiny.en.tar.bz2
fi

dart run \
  ./bin/whisper.dart \
  --encoder ./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx \
  --decoder ./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx \
  --tokens ./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt \
  --input-wav ./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav
