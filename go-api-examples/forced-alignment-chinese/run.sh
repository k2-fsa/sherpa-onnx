#!/usr/bin/env bash

set -ex

export CGO_ENABLED=1

if [ ! -d ./zh-aishell3-pinyin ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2
  tar xvf zh-aishell3-pinyin.tar.bz2
  rm zh-aishell3-pinyin.tar.bz2
fi

go mod tidy
go build

./forced-alignment-chinese
