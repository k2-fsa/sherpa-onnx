#!/usr/bin/env bash

set -ex

if [ ! -d ./sherpa-onnx-whisper-tiny.en ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
  tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
  rm sherpa-onnx-whisper-tiny.en.tar.bz2
fi

dotnet run \
  --num-threads=2 \
  --whisper-encoder=./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx \
  --whisper-decoder=./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx \
  --tokens=./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt \
  --files ./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav \
  ./sherpa-onnx-whisper-tiny.en/test_wavs/1.wav \
  ./sherpa-onnx-whisper-tiny.en/test_wavs/8k.wav
