#!/usr/bin/env bash

set -ex

repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
model=sherpa-onnx-paraformer-zh-small-2024-03-09

if [ ! -d $model ]; then
  wget $repo_url/$model.tar.bz2
  tar xvf $model.tar.bz2
  rm $model.tar.bz2
fi

cargo run --example paraformer -- \
  --model ./$model/model.int8.onnx \
  --tokens ./$model/tokens.txt \
  --wav ./$model/test_wavs/0.wav