#!/usr/bin/env bash

set -ex

export CGO_ENABLED=1

if [ ! -d ./english-mfa-v3.1.0 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2
  tar xvf english-mfa-v3.1.0.tar.bz2
  rm english-mfa-v3.1.0.tar.bz2
fi

go mod tidy
go build

./forced-alignment-english
