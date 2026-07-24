#!/usr/bin/env bash

set -ex

source ./setup.sh

if [[ ! -f "./catt_eo_model_onnx/encoder.onnx" || ! -f "./catt_eo_model_onnx/decoder.onnx"  ]]; then
  curl -SL -O https://github.com/abjadai/catt/releases/download/v2/eo_model_onnx.zip
  unzip eo_model_onnx.zip -d catt_eo_model_onnx
  rm eo_model_onnx.zip
fi

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  ./OfflineAddDiacritics.java
