#!/usr/bin/env bash

set -ex

if [ ! -d ./sherpa-onnx-paraformer-zh-2023-03-28 ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28
  cd sherpa-onnx-paraformer-zh-2023-03-28
  git lfs pull --include "*.onnx"
  cd ..
fi

dotnet run \
  --tokens=./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
  --paraformer=./sherpa-onnx-paraformer-zh-2023-03-28/model.onnx \
  --num-threads=2 \
  --files ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav \
  ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav \
  ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav \
  ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav
