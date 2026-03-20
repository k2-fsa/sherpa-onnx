#!/usr/bin/env bash
set -ex

repo=sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8
if [ ! -f ./$repo/model.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/$repo.tar.bz2
  tar xvf $repo.tar.bz2
  rm $repo.tar.bz2
fi

cargo run --example offline_punctuation --   --ct-transformer ./$repo/model.int8.onnx   --provider cpu   --num-threads 1
