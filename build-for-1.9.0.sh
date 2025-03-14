#!/usr/bin/env bash


set -ex

if [ ! -f ./setup.py || -f ./sherpa-onnx/c-api/c-api.h || ! -f ./android/SherpaOnnx ]; then
  echo "please run this script inside the sherpa-onnx directory"
  exit 1
fi

if [ ! -d /Users/fangjun/t/onnxruntime-osx-x64-1.9.0/lib ]; then
  mkdir -p /Users/fangjun/t
  pushd /Users/fangjun/t
  wget https://github.com/microsoft/onnxruntime/releases/download/v1.9.0/onnxruntime-osx-x64-1.9.0.tgz
  tar xvf onnxruntime-osx-x64-1.9.0.tgz
  rm onnxruntime-osx-x64-1.9.0.tgz
  popd
fi

export SHERPA_ONNXRUNTIME_LIB_DIR=/Users/fangjun/t/onnxruntime-osx-x64-1.9.0/lib
export SHERPA_ONNXRUNTIME_INCLUDE_DIR=/Users/fangjun/t/onnxruntime-osx-x64-1.9.0/include

mkdir -p ./build-1.9.0
cd ./build-1.9.0
cmake -DBUILD_SHARED_LIBS=ON ..
make
