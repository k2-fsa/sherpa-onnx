#!/usr/bin/env bash
set -ex

export CGO_ENABLED=1

if [ ! -d ./sherpa-onnx-spleeter-2stems-fp16 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/sherpa-onnx-spleeter-2stems-fp16.tar.bz2
  tar xjf sherpa-onnx-spleeter-2stems-fp16.tar.bz2
  rm sherpa-onnx-spleeter-2stems-fp16.tar.bz2
fi

if [ ! -f ./qi-feng-le-zh.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav
fi

go mod tidy
. "../../.github/scripts/go/replace-sherpa-onnx-go.sh"
go build

./source-separation spleeter
