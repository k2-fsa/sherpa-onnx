#!/usr/bin/env bash
set -ex

if [ ! -f ./UVR-MDX-NET-Voc_FT.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Voc_FT.onnx
fi

if [ ! -f ./qi-feng-le-zh.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav
fi

dotnet run
