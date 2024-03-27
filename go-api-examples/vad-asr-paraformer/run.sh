#!/usr/bin/env bash


if [ ! -f ./silero_vad.onnx ]; then
  curl -SL -O https://github.com/snakers4/silero-vad/blob/master/files/silero_vad.onnx
fi

if [ ! -f ./sherpa-onnx-paraformer-trilingual-zh-cantonese-en/model.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-trilingual-zh-cantonese-en.tar.bz2
  tar xvf sherpa-onnx-paraformer-trilingual-zh-cantonese-en.tar.bz2
  rm sherpa-onnx-paraformer-trilingual-zh-cantonese-en.tar.bz2
fi

go mod tidy
go build
./vad-asr-paraformer
