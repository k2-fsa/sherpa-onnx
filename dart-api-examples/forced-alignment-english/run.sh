#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -d ./english-mfa-v3.1.0 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2
  tar xvf english-mfa-v3.1.0.tar.bz2
  rm english-mfa-v3.1.0.tar.bz2
fi

dart run ./bin/main.dart
