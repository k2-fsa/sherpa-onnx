#!/usr/bin/env bash

set -ex

if [ ! -d sherpa-onnx-tdnn-yesno ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-tdnn-yesno.tar.bz2
  tar xvf sherpa-onnx-tdnn-yesno.tar.bz2
  rm sherpa-onnx-tdnn-yesno.tar.bz2
fi

go mod tidy
go build

./non-streaming-decode-files \
  --sample-rate=8000 \
  --feat-dim=23 \
  --tokens=./sherpa-onnx-tdnn-yesno/tokens.txt \
  --tdnn-model=./sherpa-onnx-tdnn-yesno/model-epoch-14-avg-2.onnx \
  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_0_1_0_0_0_1.wav
