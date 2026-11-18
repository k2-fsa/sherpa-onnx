#!/usr/bin/env bash

set -ex

if [ ! -d ./zh-aishell3-pinyin ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2
  tar xf zh-aishell3-pinyin.tar.bz2
  rm zh-aishell3-pinyin.tar.bz2
fi

dotnet run
