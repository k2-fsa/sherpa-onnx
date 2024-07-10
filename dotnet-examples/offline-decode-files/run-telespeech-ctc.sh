#!/usr/bin/env bash

set -ex

if [ ! -d sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2
  tar xvf sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2
  rm sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2
fi

dotnet run \
  --telespeech-ctc=./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/model.int8.onnx \
  --tokens=./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/tokens.txt \
  --model-type=telespeech_ctc \
  --files ./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/test_wavs/3-sichuan.wav
