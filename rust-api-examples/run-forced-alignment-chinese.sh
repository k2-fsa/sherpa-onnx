#!/usr/bin/env bash
set -ex

repo=zh-aishell3-pinyin
if [ ! -f ./$repo/lexicon.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/$repo.tar.bz2
  tar xvf $repo.tar.bz2
  rm $repo.tar.bz2
fi

cargo run --example forced_alignment_chinese --   --model-dir ./$repo   --wav ./$repo/zh.wav   --text '不要问你的国家能为你做什么，而要问你能为你的国家做什么。'
