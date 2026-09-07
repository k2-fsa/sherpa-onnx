#!/usr/bin/env bash
set -ex

export CGO_ENABLED=1

if [ ! -f ./UVR-MDX-NET-Voc_FT.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Voc_FT.onnx
fi

if [ ! -f ./qi-feng-le-zh.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav
fi

go mod tidy
. "../../.github/scripts/go/replace-sherpa-onnx-go.sh"
go build

./source-separation uvr
