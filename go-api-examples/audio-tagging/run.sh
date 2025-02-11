#!/usr/bin/env bash

if [ ! -f ./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2

  tar xvf sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
  rm sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
fi

go mod tidy
go build

./audio-tagging
